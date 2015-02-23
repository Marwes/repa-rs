#![feature(core, os, unboxed_closures)]
#![cfg_attr(test, feature(plugin))]
#![cfg_attr(test, plugin(quickcheck_macros))]
use std::fmt;
use std::iter::IntoIterator;
use std::default::Default;
use std::thread;
use std::os;
use std::cmp::PartialEq;
use std::ops::Deref;

#[cfg(test)]
extern crate quickcheck;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Z;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cons<T>(T, usize);

pub trait Shape: Clone + Sync {
    fn rank(&self) -> usize;
    fn zero_dim() -> Self;
    fn unit_dim() -> Self;
    fn add_dim(&self, other: &Self) -> Self;
    fn intersect_dim(&self, other: &Self) -> Self;
    fn size(&self) -> usize;
    fn to_index(&self, other: &Self) -> usize;
    fn from_index(&self, other: usize) -> Self;
    fn check_bounds(&self, index: &Self) -> bool;
    fn map<F>(&self, f: F) -> Self
        where F: FnMut(usize) -> usize;
}
impl Shape for Z {
    #[inline]
    fn rank(&self) -> usize { 0 }
    #[inline]
    fn zero_dim() -> Z { Z } 
    #[inline]
    fn unit_dim() -> Z { Z }
    #[inline]
    fn add_dim(&self, _: &Z) -> Z { Z }
    #[inline]
    fn intersect_dim(&self, _: &Z) -> Z { Z }
    #[inline]
    fn size(&self) -> usize { 1 }
    #[inline]
    fn to_index(&self, _: &Z) -> usize { 0 }
    #[inline]
    fn from_index(&self, _: usize) -> Z { Z }
    #[inline]
    fn check_bounds(&self, _index: &Z) -> bool { true }
    #[inline]
    fn map<F>(&self, _: F) -> Z
        where F: FnMut(usize) -> usize {
        Z
    }
}
impl <T: Shape> Shape for Cons<T> {
    #[inline]
    fn rank(&self) -> usize { self.0.rank() + 1 }
    #[inline]
    fn zero_dim() -> Cons<T> { Cons(Shape::zero_dim(), 0) } 
    #[inline]
    fn unit_dim() -> Cons<T> { Cons(Shape::unit_dim(), 1) }
    #[inline]
    fn add_dim(&self, other: &Cons<T>) -> Cons<T> {
        Cons(self.0.add_dim(&other.0), self.1 + other.1)
    }
    #[inline]
    fn intersect_dim(&self, other: &Cons<T>) -> Cons<T> {
        Cons(self.0.intersect_dim(&other.0), ::std::cmp::min(self.1, other.1))
    }
    #[inline]
    fn size(&self) -> usize { self.1 * self.0.size() }
    #[inline]
    fn to_index(&self, index: &Cons<T>) -> usize {
        self.0.to_index(&index.0) * self.1 + index.1
    }
    #[inline]
    fn from_index(&self, i: usize) -> Cons<T> {
        let r = if self.0.rank() == 0 { i } else { i % self.1 };
        Cons(self.0.from_index(i / self.1) , r)
    }
    #[inline]
    fn check_bounds(&self, index: &Cons<T>) -> bool { index.1 < self.1 && self.0.check_bounds(&index.0) }
    #[inline]
    fn map<F>(&self, mut f: F) -> Cons<T>
        where F: FnMut(usize) -> usize {
        let i = f(self.1);
        Cons(self.0.map(f), i)
    }
}

impl fmt::Display for Z {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Z") }
}

impl <S> fmt::Display for Cons<S>
    where S: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} :. {}", self.0, self.1)
    }
}

pub trait Source: Sync {
    type Element: Send + Sync;
    type Shape: Shape;

    fn extent(&self) -> &<Self as Source>::Shape;
    fn linear_index(&self, index: usize) -> <Self as Source>::Element;

    fn index(&self, index: &<Self as Source>::Shape) -> <Self as Source>::Element {
        self.linear_index(self.extent().to_index(index))
    }

    unsafe fn unsafe_index(&self, index: &<Self as Source>::Shape) -> <Self as Source>::Element {
        self.unsafe_linear_index(self.extent().to_index(index))
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> <Self as Source>::Element {
        self.index(&self.extent().from_index(index))
    }
}

impl <'a, S> Source for &'a S
    where S: Source {
    type Element = <S as Source>::Element;
    type Shape = <S as Source>::Shape;

    fn extent(&self) -> &<Self as Source>::Shape {
        (**self).extent()
    }
    fn index(&self, index: &<Self as Source>::Shape) -> <Self as Source>::Element {
        (**self).index(index)
    }
    fn linear_index(&self, index: usize) -> <Self as Source>::Element {
        (**self).linear_index(index)
    }
    unsafe fn unsafe_index(&self, index: &<Self as Source>::Shape) -> <Self as Source>::Element {
        (**self).unsafe_index(index)
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> <Self as Source>::Element {
        (**self).unsafe_linear_index(index)
    }
}


struct Iter<S> {
    index: usize,
    end: usize,
    source: S
}
impl <S> Iterator for Iter<S>
    where S: Source {
    type Item = <S as Source>::Element;
    fn next(&mut self) -> Option<<S as Source>::Element> {
        if self.index < self.end {
            let i = self.index;
            self.index += 1;
            Some(unsafe { self.source.unsafe_linear_index(i) })
        }
        else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.end - self.index, Some(self.end - self.index))
    }
}

fn iter<S>(s: &S) -> Iter<&S>
    where S: Source {
    range_iter(s, 0, s.extent().size())
}

fn range_iter<S>(s: &S, start: usize, end: usize) -> Iter<&S>
    where S: Source {
    assert!(start <= end);
    assert!(end <= s.extent().size());
    Iter { index: start, end: end, source: s }
}

struct Fmt<S>(S);
impl <S> fmt::Display for Fmt<S>
    where S: Source
        , <S as Source>::Element: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "["));
        for e in iter(&self.0) {
            try!(write!(f, "{}, ", e));
        }
        write!(f, "]")
    }
}

pub struct UArray<S, V>
    where S: Shape
        , V: Send + Sync {
    shape: S,
    elems: V
}

impl <E, S, V> UArray<S, V>
    where S: Shape
        , E: Clone + Send + Sync
        , V: Deref<Target=[E]> + Send + Sync {

    pub fn new(shape: S, elems: V) -> UArray<S, V> {
        UArray { shape: shape, elems: elems }
    }
}

impl <E, S> UArray<S, Vec<E>>
    where S: Shape
        , E: Clone + Send + Sync {

    pub fn from_iter<I>(shape: S, iter: I) -> UArray<S, Vec<E>>
        where I: IntoIterator<Item=E> {
        UArray { shape: shape, elems: iter.into_iter().collect() }
    }
}

impl <E, S, V> Source for UArray<S, V>
    where S: Shape
        , E: Clone + Send + Sync
        , V: Deref<Target=[E]> + Send + Sync {
    type Element = E;
    type Shape = S;

    fn extent(&self) -> &S {
        &self.shape
    }
    fn linear_index(&self, index: usize) -> E {
        self.elems[index].clone()
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> <Self as Source>::Element {
        self.elems.get_unchecked(index).clone()
    }
}

impl <E, O, S, V> PartialEq<O> for UArray<S, V>
    where O: Source<Shape=S, Element=E>
        , S: Shape + PartialEq
        , V: Deref<Target=[E]> + Send + Sync
        , E: Clone + Send + Sync + PartialEq {
    fn eq(&self, other: &O) -> bool {
        if self.extent() != other.extent() {
            false
        }
        else {
            iter(self).zip(iter(other)).all(|(l, r)| l == r)
        }
    }
}

pub struct DArray<S, F>
    where S: Shape {
    shape: S,
    f: F
}

impl <S, F, E: Send + Sync> Source for DArray<S, F>
    where S: Shape
        , F: for<'a> Fn(&'a S) -> E + Sync {
    type Element = E;
    type Shape = S;

    fn extent(&self) -> &<Self as Source>::Shape {
        &self.shape
    }
    fn index(&self, index: &<Self as Source>::Shape) -> E {
        if !self.extent().check_bounds(index) {
            panic!("Array out of bounds")
        }
        (self.f)(index)
    }
    fn linear_index(&self, index: usize) -> E {
        self.index(&self.shape.from_index(index))
    }
    unsafe fn unsafe_index(&self, index: &<Self as Source>::Shape) -> E {
        (self.f)(index)
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> E {
        self.unsafe_index(&self.shape.from_index(index))
    }
}

impl <E, O, S, F> PartialEq<O> for DArray<S, F>
    where O: Source<Shape=S, Element=E>
        , S: Shape + PartialEq
        , F: Fn(&S) -> E + Sync
        , E: Clone + Send + Sync + PartialEq {
    fn eq(&self, other: &O) -> bool {
        if self.extent() != other.extent() {
            false
        }
        else {
            iter(self).zip(iter(other)).all(|(l, r)| l == r)
        }
    }
}

pub fn from_function<S, F, B>(shape: S, f: F) -> DArray<S, F>
    where F: Fn(&S) -> B, S: Shape {
    DArray {
        shape: shape,
        f: f
    }
}

pub struct MapFn<S, F> {
    source: S,
    f: F
}

impl <'a, S, A, B, F> Fn<(&'a <S as Source>::Shape,)> for MapFn<S, F>
    where A: Send + Sync
        , F: Fn(A) -> B
        , S: Source<Element=A> {
    type Output = B;
    extern "rust-call" fn call(&self, (sh,): (&<S as Source>::Shape,)) -> B {
        let e = unsafe { self.source.unsafe_index(sh) };
        (self.f)(e)
    }
}

pub fn map<S, F, B>(f: F, array: S) -> DArray<<S as Source>::Shape, MapFn<S, F>>
    where F: Fn(<S as Source>::Element) -> B, S: Source {
    DArray {
        shape: array.extent().clone(),
        f: MapFn { source: array, f: f }
    }
}

pub struct ExtractFn<S>
    where S: Source {
    source: S,
    start: <S as Source>::Shape,
}

impl <'a, S> Fn<(&'a <S as Source>::Shape,)> for ExtractFn<S>
    where S: Source {
    type Output = <S as Source>::Element;
    extern "rust-call" fn call(&self, (sh,): (&<S as Source>::Shape,)) -> <S as Source>::Element {
        let i = self.start.add_dim(sh);
        unsafe { self.source.unsafe_index(&i) }
    }
}

pub fn extract<S>(start: <S as Source>::Shape, size: <S as Source>::Shape, array: S) -> DArray<<S as Source>::Shape, ExtractFn<S>>
    where S: Source {
    if !array.extent().map(|i| i + 1).check_bounds(&start.add_dim(&size)) {
        panic!("extract: out of bounds")
    }
    DArray {
        shape: size,
        f: ExtractFn { source: array, start: start }
    }
}

pub struct TransposeFn<S>
    where S: Source {
    source: S
}

impl <'a, S, Sh> Fn<(&'a <S as Source>::Shape,)> for TransposeFn<S>
    where S: Source<Shape=Cons<Cons<Sh>>>
        , Sh: Shape {
    type Output = <S as Source>::Element;
    extern "rust-call" fn call(&self, (sh,): (&<S as Source>::Shape,)) -> <S as Source>::Element {
        let &Cons(Cons(ref rest, x), y) = sh;
        unsafe { self.source.unsafe_index(&Cons(Cons(rest.clone(), y), x)) }
    }
}

pub fn transpose<S, Sh>(array: S) -> DArray<<S as Source>::Shape, TransposeFn<S>>
    where S: Source<Shape=Cons<Cons<Sh>>>
        , Sh: Shape {
    DArray {
        shape: array.extent().clone(),
        f: TransposeFn { source: array }
    }
}

pub struct ZipWithFn<S1, S2, F>
    where S1: Source
        , S2: Source {
    lhs: S1,
    rhs: S2,
    f: F
}

impl <'a, S1, S2, Sh, F, O> Fn<(&'a <S1 as Source>::Shape,)> for ZipWithFn<S1, S2, F>
    where Sh: Shape
        , S1: Source<Shape=Sh>
        , S2: Source<Shape=Sh>
        , F: Fn(<S1 as Source>::Element, <S2 as Source>::Element) -> O {
    type Output = O;
    extern "rust-call" fn call(&self, (sh,): (&Sh,)) -> O {
        unsafe {
            let l = self.lhs.unsafe_index(sh);
            let r = self.rhs.unsafe_index(sh);
            (self.f)(l, r)
        }
    }
}

pub fn zip_with<S1, S2, F, O>(lhs: S1, rhs: S2, f: F) -> DArray<<S1 as Source>::Shape, ZipWithFn<S1, S2, F>>
    where S1: Source
        , S2: Source<Shape=<S1 as Source>::Shape>
        , F: Fn(<S1 as Source>::Element, <S2 as Source>::Element) -> O {
    DArray {
        shape: lhs.extent().intersect_dim(rhs.extent()),
        f: ZipWithFn { lhs: lhs, rhs: rhs, f: f }
    }
}

pub struct TraverseFn<S, T>
    where S: Source {
    source: S,
    transform: T
}

impl <'a, S, Sh, T, B> Fn<(&'a Sh,)> for TraverseFn<S, T>
    where Sh: Shape
        , S: Source
        , T: for<'b, 'c> Fn(&'b S, &'c Sh) -> B {
    type Output = B;
    extern "rust-call" fn call(&self, (sh,): (&Sh,)) -> B {
        (self.transform)(&self.source, sh)
    }
}

pub fn traverse<S, Sh, F, T, A, B>(array: S, new_shape: F, transform: T) -> DArray<Sh, TraverseFn<S, T>>
    where S: Source
        , Sh: Shape
        , F: FnOnce(&<S as Source>::Shape) -> Sh
        , T: Fn(&S, &Sh) -> B {

    let shape = new_shape(array.extent());
    DArray {
        shape: shape,
        f: TraverseFn { source: array, transform: transform }
    }
}


pub fn fold_s<F, S, Sh>(mut f: F, e: <S as Source>::Element, array: &S) -> UArray<Sh, Vec<<S as Source>::Element>>
    where Sh: Shape
        , S: Source<Shape=Cons<Sh>>
        , F: FnMut(<S as Source>::Element, <S as Source>::Element) -> <S as Source>::Element
        , <S as Source>::Element: Clone {
    let &Cons(ref shape, size) = array.extent();
    let new_size = shape.size();
    let mut elems = Vec::with_capacity(new_size);
    for offset in (0..new_size).map(|i| i * size) {
        let mut r = e.clone();
        for e in range_iter(array, offset, offset + size) {
            r = f(r, e);
        }
        elems.push(r);
    }
    UArray {
        shape: shape.clone(),
        elems: elems
    }
}

pub fn fold_p<F, S, Sh>(ref f: F, ref e: <S as Source>::Element, array: &S) -> UArray<Sh, Vec<<S as Source>::Element>>
    where Sh: Shape
        , S: Source<Shape=Cons<Sh>>
        , F: Fn(<S as Source>::Element, <S as Source>::Element) -> <S as Source>::Element + Sync
        , <S as Source>::Element: Clone {
    let &Cons(ref shape, size) = array.extent();
    let new_size = shape.size();
    let mut elems = vec![e.clone(); new_size];
    parallel_chunks(&mut elems, |chunk, start| {
        let end = start + chunk.len();
        for (chunk, offset) in chunk.iter_mut().zip((start..end).map(|i| i * size)) {
            let mut r = e.clone();
            for e in range_iter(array, offset, (offset + size)) {
                r = f(r, e);
            }
            *chunk = r;
        }
    });
    UArray {
        shape: shape.clone(),
        elems: elems
    }
}

pub fn compute_s<S>(array: &S) -> UArray<<S as Source>::Shape, Vec<<S as Source>::Element>>
    where S: Source {
    UArray {
        shape: array.extent().clone(),
        elems: iter(array).collect()
    }
}

pub fn compute_p<S>(array: &S) -> UArray<<S as Source>::Shape, Vec<<S as Source>::Element>>
    where S: Source
        , <S as Source>::Element: Default + Clone {
    let extent = array.extent();
    let size = extent.size();
    let mut elems = vec![Default::default(); size];
    parallel_chunks(&mut elems, |chunk, offset| {
        let len = chunk.len();
        for (r, e) in chunk.iter_mut().zip(range_iter(array, offset, offset + len)) {
            *r = e;
        }
    });
    UArray {
        shape: extent.clone(),
        elems: elems
    }
}

fn parallel_chunks<E, F>(elems: &mut [E], ref f: F)
    where E: Send + Sync
        , F: Fn(&mut [E], usize) + Sync {
    let len = elems.len();
    let slices = os::num_cpus();
    let slice_len = (len + slices - 1) / slices;
    //Save the join guards so that they are 
    let mut results = Vec::new();
    for (i, chunk) in elems.chunks_mut(slice_len).enumerate() {
        let offset = i * slice_len;
        let x = thread::scoped(move || f(chunk, offset));
        results.push(x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;
    
    
    const SHAPE2X2: Cons<Cons<Z>> = Cons(Cons(Z, 2), 2);

    #[test]
    fn index() {
        let i_0_0 = Cons(Cons(Z, 0), 0);
        assert_eq!(SHAPE2X2.to_index(&i_0_0), 0);
    }
    
    #[test]
    fn function() {
        let m = from_function(SHAPE2X2, |i| SHAPE2X2.to_index(i));
        assert_eq!(m.linear_index(3), 3);
        assert_eq!(m.index(&SHAPE2X2.from_index(3)), 3);
    }

    #[test]
    fn array_index() {
        let matrix = vec![1, 2
                        , 3, 4];
        let array = UArray::new(SHAPE2X2, matrix);
        assert_eq!(array.index(&Cons(Cons(Z, 0), 0)), 1);
        let delayed = map(|x| x * 2, &array);
        assert_eq!(delayed.index(&Cons(Cons(Z, 0), 0)), 2);
        assert_eq!(delayed.index(&Cons(Cons(Z, 1), 0)), 6);
    }

    #[test]
    fn extract_test() {
        let m = from_function(SHAPE2X2, |i| SHAPE2X2.to_index(i));
        let row2 = extract(Cons(Cons(Z, 1), 0), Cons(Cons(Z, 1), 2), &m);
        assert_eq!(row2.index(&Cons(Cons(Z, 0), 0)), 2);
        assert_eq!(row2.index(&Cons(Cons(Z, 0), 1)), 3);
    }

    #[test]
    fn compute_parallel() {
        let size = Cons(Cons(Z, 1000), 1000);
        let m = from_function(size.clone(), |i| size.to_index(i));
        let indexes = compute_p(&m);
        let i = Cons(Cons(Z, 200), 21);
        assert_eq!(indexes.index(&i), indexes.extent().to_index(&i));
    }

    #[test]
    fn zip_with_test() {
        let m = from_function(SHAPE2X2, |i| SHAPE2X2.to_index(i));
        let m2 = zip_with(&m, &m, |l, r| l + r);
        let i0x1 = Cons(Cons(Z, 0), 1);
        let i1x0 = Cons(Cons(Z, 1), 0);
        assert_eq!(m2.index(&i0x1), 2);
        assert_eq!(m2.index(&i1x0), 4);
    }

    #[test]
    fn fold_sequential() {
        let m = from_function(SHAPE2X2, |i| SHAPE2X2.to_index(i));
        let m2 = fold_s(|l, r| l + r, 0, &m);
        assert_eq!(m2.index(&Cons(Z, 0)), 1);
        assert_eq!(m2.index(&Cons(Z, 1)), 5);
    }

    #[test]
    fn fold_parallel() {
        let m = from_function(Cons(Z, 10000), |i| Cons(Z, 10000).to_index(i));
        let m2 = fold_p(|l, r| l + r, 0, &m);
        assert_eq!(m2.index(&Z), (9999 + 0) * 10000 / 2);
    }

    #[quickcheck]
    fn to_index_from_index(s: (usize, usize, usize), i: (usize, usize, usize)) -> TestResult {
        let shape = Cons(Cons(Cons(Z, s.0), s.1), s.2);
        let index = Cons(Cons(Cons(Z, i.0), i.1), i.2);
        if !shape.check_bounds(&index) {
            return TestResult::discard();
        }
        TestResult::from_bool(shape.from_index(shape.to_index(&index)) == index)
    }

    #[quickcheck]
    fn transpose_test(vs: Vec<i32>) -> bool {
        let size = ::std::num::Float::sqrt(vs.len() as f64) as usize;
        let m = UArray::new(Cons(Cons(Z, size), size), vs);
        transpose(transpose(&m)) == m
    }

}
