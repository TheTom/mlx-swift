// Copyright © 2024 Apple Inc.

import Foundation

/// F-83: Maximum number of elements for stack-allocated Int32 conversion buffers.
/// Shapes and axes in transformer ops almost never exceed 8 dimensions, so the
/// fast path covers ~all decode-step calls (reshape, transpose, softmax, squeeze,
/// sum, expand_dims, broadcast_to). Cherry-picked from ekryski 29525f6.
@usableFromInline
let _maxStackInt32 = 8

extension [Int] {

    /// Convenience to coerce array of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt32: [Int32] {
        self.map { Int32($0) }
    }

    /// Convenience to coerce array of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt64: [Int64] {
        self.map { Int64($0) }
    }

    /// F-83 stack-allocated Int→Int32 conversion. Eliminates the per-op heap
    /// allocation that `asInt32` does for shape/axes arrays at every reshape /
    /// transpose / softmax / squeeze / sum call. The buffer is only valid for
    /// `body`'s scope. Falls back to heap for arrays > `_maxStackInt32` elements.
    @inlinable
    func withInt32Buffer<R>(_ body: (UnsafeBufferPointer<Int32>) throws -> R) rethrows -> R {
        let n = count
        if n <= _maxStackInt32 {
            return try withUnsafeTemporaryAllocation(
                of: Int32.self, capacity: _maxStackInt32
            ) { buf in
                for i in 0..<n {
                    buf[i] = Int32(self[i])
                }
                return try body(UnsafeBufferPointer(start: buf.baseAddress, count: n))
            }
        } else {
            let converted = self.asInt32
            return try converted.withUnsafeBufferPointer(body)
        }
    }
}

extension Sequence<Int> {

    /// Convenience to coerce  sequence of `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var asInt32: [Int32] {
        self.map { Int32($0) }
    }

    @inlinable
    var asInt64: [Int64] {
        self.map { Int64($0) }
    }

    /// F-83 stack-allocated Int→Int32 conversion for arbitrary Int sequences.
    /// Mirrors the `[Int].withInt32Buffer` overload but works on `some Collection<Int>`
    /// parameters in Ops without forcing the caller to `Array(...)`-wrap.
    @inlinable
    func withInt32Buffer<R>(_ body: (UnsafeBufferPointer<Int32>) throws -> R) rethrows -> R {
        let arr = Array(self)
        return try arr.withInt32Buffer(body)
    }
}

extension Int {

    /// Convenience to convert `Int` to `Int32` -- Cmlx uses `Int32` for many things but it is
    /// more natural to use `Int` in Swift.
    @inlinable
    var int32: Int32 { Int32(self) }

    @inlinable
    var int64: Int64 { Int64(self) }
}
