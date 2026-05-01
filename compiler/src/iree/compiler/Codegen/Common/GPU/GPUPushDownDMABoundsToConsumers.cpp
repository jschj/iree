// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===-- GPUPushDownDMABoundsToConsumers.cpp -------------------------------===//
//
// For each iree_gpu.coalesced_gather_dma whose innermost in_bounds entry is
// false, inserts a tensor.extract_slice + tensor.pad chain between the DMA's
// outer scf.forall result and its consumer. Downstream vectorization
// (enable-vector-masking=true) then lowers the pad to a masked
// vector.transfer_read via vectorizeAsTensorPadOp, discarding straddle-
// corrupted columns and substituting zero.
//
// See docs/superpowers/specs/2026-04-29-dma-mask-pushdown-design.md.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUPUSHDOWNDMABOUNDSTOCONSUMERSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// Walk from the DMA's init operand (a forall sharedOut block argument) out
// through the chain of nested forall shared_outs and parallel_insert_slices
// until we reach an scf.forall whose result is read by a non-parallel_insert
// consumer. That result is the SSA value the matmul (or other consumer) sees.
//
// Two link types are followed:
//   1. BlockArgument (forall sharedOut) -> forall.getResult(idx)
//      (existing nested-forall handling)
//   2. forall result that is the source of a tensor.parallel_insert_slice
//      in a parent forall's terminator -> the parent forall's destination
//      sharedOut (a BlockArgument), which feeds back into case (1).
//
// The walk stops as soon as we hit a value whose only forward chain breaks
// (no enclosing forall, no propagating parallel_insert_slice, etc.).
static Value walkUpSharedOuts(Value v) {
  while (true) {
    if (auto bbarg = dyn_cast<BlockArgument>(v)) {
      auto forall =
          dyn_cast<scf::ForallOp>(bbarg.getOwner()->getParentOp());
      if (!forall)
        return v;
      unsigned argIdx = bbarg.getArgNumber();
      unsigned sharedOutsStart = forall.getRank();
      if (argIdx < sharedOutsStart)
        return v;
      v = forall.getResult(argIdx - sharedOutsStart);
      continue;
    }

    // v is an SSA value (typically an scf.forall result). If it's only
    // consumed by a parallel_insert_slice inside a parent forall, follow
    // that link to the parent forall's sharedOut BlockArg.
    auto definingForall = v.getDefiningOp<scf::ForallOp>();
    if (!definingForall)
      return v;
    auto parentForall = definingForall->getParentOfType<scf::ForallOp>();
    if (!parentForall)
      return v;

    Value nextSharedOut = nullptr;
    for (Operation &op : parentForall.getTerminator().getRegion().front()) {
      auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(&op);
      if (!insert)
        continue;
      if (insert.getSource() == v) {
        nextSharedOut = insert.getDest();
        break;
      }
    }
    if (!nextSharedOut)
      return v;
    v = nextSharedOut;
  }
}

// When the DMA source's innermost row size in bytes is not DWORD (4-byte)
// aligned, the AMD HW partial-DWORD OOB clamp zeroes valid bytes at the
// buffer end. Wrap the source with an iree_gpu.buffer_resource_cast that
// declares validBytes rounded up to the next multiple of 4. After
// bufferization this becomes amdgpu.fat_raw_buffer_cast with the explicit
// validBytes, which keeps the trailing partial DWORD in-bounds. The garbage
// in bytes [naturalBytes, roundedBytes) lands in masked LDS columns and is
// discarded by the consumer-side tensor.pad inserted below.
//
// Returns failure when no rewrite is needed (already aligned, dynamic
// element bit width, etc.). Source is mutated in place.
static LogicalResult padSourceBufferDescriptorToDWORD(
    IRRewriter &rewriter, IREE::GPU::CoalescedGatherDMAOp dma) {
  Value src = dma.getSource();
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  if (!srcTy)
    return failure();
  Type elemTy = srcTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return failure();
  unsigned elemBits = elemTy.getIntOrFloatBitWidth();
  if (elemBits == 0 || elemBits % 8 != 0)
    return failure();
  unsigned elemBytes = elemBits / 8;

  // If the innermost row is statically DWORD-aligned, the partial-DWORD
  // straddle issue cannot occur and we don't need to pad the descriptor.
  unsigned rank = srcTy.getRank();
  int64_t innermostStatic = srcTy.getDimSize(rank - 1);
  if (!ShapedType::isDynamic(innermostStatic) &&
      (innermostStatic * elemBytes) % 4 == 0) {
    return failure();
  }

  // The DMA source can be a per-K-block tensor.extract_slice of a larger
  // tensor (e.g. matmul tiled by a reduction dim). The buffer descriptor's
  // validBytes must reflect the *underlying* allocation, not the slice —
  // sizing it from the slice would clamp the descriptor below the size that
  // earlier accesses already need. Walk through extract_slices to the root.
  Value root = src;
  while (auto sl = root.getDefiningOp<tensor::ExtractSliceOp>()) {
    root = sl.getSource();
  }
  auto rootTy = dyn_cast<RankedTensorType>(root.getType());
  if (!rootTy)
    return failure();
  unsigned rootRank = rootTy.getRank();

  // If a buffer_resource_cast with valid_bytes already wraps the root, skip.
  if (auto existing = root.getDefiningOp<IREE::GPU::BufferResourceCastOp>()) {
    if (existing.getValidBytes())
      return failure();
  }

  // Compute valid_bytes = roundUp(prod(rootDim_i) * elemBytes + 4, 4).
  // Place the cast in a scope that dominates the DMA but lives outside any
  // enclosing forall so the validBytes computation is hoisted out of loops.
  if (auto *rootOp = root.getDefiningOp()) {
    rewriter.setInsertionPointAfter(rootOp);
  } else {
    // Block argument: insert at the start of its block (e.g. function entry).
    auto *block = cast<BlockArgument>(root).getOwner();
    rewriter.setInsertionPointToStart(block);
  }
  Location loc = dma.getLoc();

  MLIRContext *ctx = rewriter.getContext();
  SmallVector<OpFoldResult> dims;
  // Track ops we create that legitimately need to keep reading `root` (so
  // they don't get redirected to the cast we're about to insert, which would
  // then violate dominance — the cast lives below them in the block).
  SmallPtrSet<Operation *, 4> preserveRootUsers;
  AffineExpr prod = getAffineConstantExpr(elemBytes, ctx);
  for (unsigned d = 0; d < rootRank; ++d) {
    int64_t s = rootTy.getDimSize(d);
    if (ShapedType::isDynamic(s)) {
      auto dimOp = tensor::DimOp::create(rewriter, loc, root, d);
      preserveRootUsers.insert(dimOp);
      dims.push_back(OpFoldResult(dimOp.getResult()));
      prod = prod * getAffineSymbolExpr(dims.size() - 1, ctx);
    } else {
      prod = prod * getAffineConstantExpr(s, ctx);
    }
  }
  // We must guarantee the *final* DWORD a lane reads past the natural end is
  // covered. Adding 4 then rounding up to DWORD ensures at least a 4-byte
  // safety margin even when naturalBytes happens to be DWORD-aligned (e.g.
  // 64x43 f16 = 5504 bytes; the last lane's straddle reads bytes 5502..5505,
  // requiring validBytes >= 5508).
  AffineExpr rounded =
      (prod + getAffineConstantExpr(7, ctx)).floorDiv(4) * 4;
  AffineMap map = AffineMap::get(/*dimCount=*/0,
                                 /*symbolCount=*/dims.size(), rounded);
  OpFoldResult validBytesOFR =
      affine::makeComposedFoldedAffineApply(rewriter, loc, map, dims);
  Value validBytes =
      getValueOrCreateConstantIndexOp(rewriter, loc, validBytesOFR);

  auto castOp = IREE::GPU::BufferResourceCastOp::create(
      rewriter, loc, rootTy, root,
      /*cache_swizzle_stride=*/Value{},
      /*valid_bytes=*/validBytes);

  // Re-route uses of `root` to the cast. Keep the cast itself plus any
  // tensor.dim ops we created above (they were emitted *before* the cast and
  // need the original `root`; redirecting them would violate dominance).
  preserveRootUsers.insert(castOp);
  rewriter.replaceAllUsesExcept(root, castOp.getResult(), preserveRootUsers);
  return success();
}

// Insert extract_slice + tensor.pad after the outermost forall for one DMA.
// Returns failure if the DMA doesn't match the expected pattern (no-op).
static LogicalResult rewriteOneDMA(IRRewriter &rewriter,
                                   IREE::GPU::CoalescedGatherDMAOp dma) {
  // Only act when innermost dimension is out-of-bounds.
  std::optional<ArrayAttr> inBoundsOpt = dma.getInBounds();
  if (!inBoundsOpt || inBoundsOpt->empty())
    return failure();
  ArrayAttr inBounds = *inBoundsOpt;
  unsigned innermost = inBounds.size() - 1;
  if (cast<BoolAttr>(inBounds[innermost]).getValue())
    return failure(); // innermost is in-bounds; nothing to do

  // Resolve the LDS tile type from the init operand.
  auto tileTy = dyn_cast<RankedTensorType>(dma.getInit().getType());
  if (!tileTy)
    return failure();
  unsigned rank = tileTy.getRank();
  int64_t innerTileSize = tileTy.getDimSize(innermost);
  if (ShapedType::isDynamic(innerTileSize))
    return failure(); // shouldn't occur; bail defensively

  // Find the outermost forall result visible to consumers.
  Value outerResult = walkUpSharedOuts(dma.getInit());
  auto outerForall =
      dyn_cast_or_null<scf::ForallOp>(outerResult.getDefiningOp());
  if (!outerForall)
    return failure();

  // Insert after the outermost forall.
  rewriter.setInsertionPointAfter(outerForall);
  Location loc = dma.getLoc();

  // tensor.dim of the DMA source for the innermost dimension.
  Value src = dma.getSource();
  Value innerExtent = tensor::DimOp::create(rewriter, loc, src, innermost);
  Value innerTileV = arith::ConstantIndexOp::create(rewriter, loc, innerTileSize);
  Value padAmount = arith::SubIOp::create(rewriter, loc, innerTileV, innerExtent);

  // tensor.extract_slice: cut the valid columns out of the LDS tile.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> validSizes;
  for (unsigned d = 0; d < rank; ++d) {
    if (d == innermost)
      validSizes.push_back(OpFoldResult(innerExtent));
    else
      validSizes.push_back(rewriter.getIndexAttr(tileTy.getDimSize(d)));
  }
  Value valid = tensor::ExtractSliceOp::create(rewriter, loc, outerResult,
                                               offsets, validSizes, strides);

  // tensor.pad: re-expand to the full tile shape with zero padding.
  SmallVector<OpFoldResult> lowPad(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> highPad(rank, rewriter.getIndexAttr(0));
  highPad[innermost] = OpFoldResult(padAmount);

  Type elemTy = tileTy.getElementType();
  TypedAttr zeroAttr = rewriter.getZeroAttr(elemTy);
  if (!zeroAttr)
    return failure();
  Value padCst = arith::ConstantOp::create(rewriter, loc, zeroAttr);

  auto padOp = tensor::PadOp::create(rewriter, loc, tileTy, valid, lowPad,
                                     highPad, /*nofold=*/false);
  Block *body = rewriter.createBlock(&padOp.getBodyRegion());
  for (unsigned i = 0; i < rank; ++i)
    body->addArgument(rewriter.getIndexType(), loc);
  rewriter.setInsertionPointToStart(body);
  tensor::YieldOp::create(rewriter, loc, padCst);
  rewriter.setInsertionPointAfter(padOp);

  // Replace all uses of the forall result with the re-padded tensor, except
  // for the extract_slice we just created which must read the original.
  outerResult.replaceAllUsesExcept(padOp.getResult(),
                                   valid.getDefiningOp<tensor::ExtractSliceOp>());
  return success();
}

struct GPUPushDownDMABoundsToConsumersPass final
    : impl::GPUPushDownDMABoundsToConsumersPassBase<
          GPUPushDownDMABoundsToConsumersPass> {
  using Base::Base;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<IREE::GPU::CoalescedGatherDMAOp> dmas;
    funcOp.walk(
        [&](IREE::GPU::CoalescedGatherDMAOp dma) { dmas.push_back(dma); });

    for (auto dma : dmas) {
      // Best-effort buffer-descriptor padding for non-DWORD-aligned sources.
      // Independent of the consumer-side pad rewrite below.
      (void)padSourceBufferDescriptorToDWORD(rewriter, dma);
      (void)rewriteOneDMA(rewriter, dma);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
