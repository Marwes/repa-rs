#![feature(core, os, unboxed_closures)]
use std::fmt;
use std::iter::IntoIterator;
use std::default::Default;
use std::thread;
use std::os;

#[derive(Clone, Debug)]
pub struct Z;

#[derive(Clone, Debug)]
pub struct Cons<T>(T, usize);

pub trait Shape: Clone + Sync {
    fn rank(&self) -> usize;
    fn zero_dim() -> Self;
    fn unit_dim() -> Self;
    fn add_dim(&self, other: &Self) -> Self;
    fn size(&self) -> usize;
    fn to_index(&self, other: &Self) -> usize;
    fn from_index(&self, other: usize) -> Self;
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
    fn size(&self) -> usize { 1 }
    #[inline]
    fn to_index(&self, _: &Z) -> usize { 0 }
    #[inline]
    fn from_index(&self, _: usize) -> Z { Z }
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
}


pub trait Source: Sync {
    type Element: Send + Sync;
    type Sh: Shape;

    fn extent(&self) -> &<Self as Source>::Sh;
    fn linear_index(&self, index: usize) -> <Self as Source>::Element;

    fn index(&self, index: &<Self as Source>::Sh) -> <Self as Source>::Element {
        self.linear_index(self.extent().to_index(index))
    }

    unsafe fn unsafe_index(&self, index: &<Self as Source>::Sh) -> <Self as Source>::Element {
        self.unsafe_linear_index(self.extent().to_index(index))
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> <Self as Source>::Element {
        self.index(&self.extent().from_index(index))
    }
}

impl <'a, S> Source for &'a S
    where S: Source {
    type Element = <S as Source>::Element;
    type Sh = <S as Source>::Sh;

    fn extent(&self) -> &<Self as Source>::Sh {
        (**self).extent()
    }
    fn index(&self, index: &<Self as Source>::Sh) -> <Self as Source>::Element {
        (**self).index(index)
    }
    fn linear_index(&self, index: usize) -> <Self as Source>::Element {
        (**self).linear_index(index)
    }
    unsafe fn unsafe_index(&self, index: &<Self as Source>::Sh) -> <Self as Source>::Element {
        (**self).unsafe_index(index)
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> <Self as Source>::Element {
        (**self).unsafe_linear_index(index)
    }
}


struct Iter<S> {
    index: usize,
    source: S
}
impl <S> Iterator for Iter<S>
    where S: Source {
    type Item = <S as Source>::Element;
    fn next(&mut self) -> Option<<S as Source>::Element> {
        if self.index < self.source.extent().size() {
            let i = self.index;
            self.index += 1;
            Some(self.source.linear_index(i))
        }
        else {
            None
        }
    }
}
fn iter<S>(s: &S) -> Iter<&S>
    where S: Source {
    Iter { index: 0, source: s }
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

pub struct UArray<S, E> {
    shape: S,
    elems: Vec<E>
}

impl <S: Shape, E: Clone> UArray<S, E> {
    pub fn new(shape: S, e: E) -> UArray<S, E> {
        let size = shape.size();
        let elems = vec![e; size];
        UArray { shape: shape, elems: elems }
    }
    pub fn from_iter<I>(shape: S, iter: I) -> UArray<S, E>
        where I: IntoIterator<Item=E> {
        UArray { shape: shape, elems: iter.into_iter().collect() }
    }
}

impl <E: Copy + Send + Sync, S: Shape> Source for UArray<S, E> {
    type Element = E;
    type Sh = S;

    fn extent(&self) -> &S {
        &self.shape
    }
    fn linear_index(&self, index: usize) -> E {
        self.elems[index]
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> <Self as Source>::Element {
        *self.elems.get_unchecked(index)
    }
}

trait UnsafeFn<Args> {
    type Output;
    unsafe fn unsafe_call(&self, args: Args) -> <Self as UnsafeFn<Args>>::Output { self.safe_call(args) }
    fn safe_call(&self, args: Args) -> <Self as UnsafeFn<Args>>::Output;
}
impl <F, Args> UnsafeFn<Args> for F
    where F: Fn<Args> {
    type Output = <F as Fn<Args>>::Output;
    fn safe_call(&self, args: Args) -> <Self as UnsafeFn<Args>>::Output {
        self.call(args)
    }
}

pub struct DArray<S, F>
    where S: Shape {
    shape: S,
    f: F
}

impl <S, F, E: Send + Sync> Source for DArray<S, F>
    where S: Shape
        , F: for<'a> UnsafeFn(&'a S) -> E + Sync {
    type Element = E;
    type Sh = S;

    fn extent(&self) -> &<Self as Source>::Sh {
        &self.shape
    }
    fn index(&self, index: &<Self as Source>::Sh) -> E {
        self.f.safe_call((index,))
    }
    fn linear_index(&self, index: usize) -> E {
        self.index(&self.shape.from_index(index))
    }
    unsafe fn unsafe_index(&self, index: &<Self as Source>::Sh) -> E {
        self.f.unsafe_call((index,))
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> E {
        self.unsafe_index(&self.shape.from_index(index))
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

impl <'a, S, A, B, F> UnsafeFn<(&'a <S as Source>::Sh,)> for MapFn<S, F>
    where A: Send + Sync
        , F: Fn(A) -> B
        , S: Source<Element=A> {
    type Output = B;
    unsafe fn unsafe_call(&self, (sh,): (&<S as Source>::Sh,)) -> B {
        let e = self.source.unsafe_index(sh);
        (self.f)(e)
    }
    fn safe_call(&self, (sh,): (&<S as Source>::Sh,)) -> B {
        let e = self.source.index(sh);
        (self.f)(e)
    }
}

pub fn map<S, F, B>(f: F, array: &S) -> DArray<<S as Source>::Sh, MapFn<&S, F>>
    where F: Fn(<S as Source>::Element) -> B, S: Source {
    DArray {
        shape: array.extent().clone(),
        f: MapFn { source: array, f: f }
    }
}

pub struct ExtractFn<S>
    where S: Source {
    source: S,
    start: <S as Source>::Sh,
}

impl <'a, S> UnsafeFn<(&'a <S as Source>::Sh,)> for ExtractFn<S>
    where S: Source {
    type Output = <S as Source>::Element;
    unsafe fn unsafe_call(&self, (sh,): (&<S as Source>::Sh,)) -> <S as Source>::Element {
        let i = self.start.add_dim(sh);
        self.source.unsafe_index(&i)
    }
    fn safe_call(&self, (sh,): (&<S as Source>::Sh,)) -> <S as Source>::Element {
        let i = self.start.add_dim(sh);
        self.source.index(&i)
    }
}

pub fn extract<S>(start: <S as Source>::Sh, size: <S as Source>::Sh, array: &S) -> DArray<<S as Source>::Sh, ExtractFn<&S>>
    where S: Source {
    DArray {
        shape: size.clone(),
        f: ExtractFn { source: array, start: start }
    }
}

pub fn compute_s<S>(array: &S) -> UArray<<S as Source>::Sh, <S as Source>::Element>
    where S: Source {
    let size = array.extent().size();
    let mut elems = Vec::with_capacity(size);
    for i in 0..size {
        elems.push(array.linear_index(i));
    }
    UArray {
        shape: array.extent().clone(),
        elems: elems
    }
}
pub fn compute_p<S>(array: &S) -> UArray<<S as Source>::Sh, <S as Source>::Element>
    where S: Source
        , <S as Source>::Element: Default + Clone {
    let size = array.extent().size();
    let mut elems = vec![Default::default(); size];
    {
        let len = elems.len();
        let slices = os::num_cpus();
        let slice_len = len / slices;
        //Save the join guards so that they are 
        let mut results = Vec::new();
        for (i, chunk) in elems.chunks_mut(slice_len).enumerate() {
            let offset = i * slice_len;
            let x = thread::scoped(move || {
                for (j, e) in chunk.iter_mut().enumerate() {
                    *e = array.linear_index(offset + j);
                }
            });
            results.push(x);
        }
        for r in results {
            r.join();
        }
    }
    UArray {
        shape: array.extent().clone(),
        elems: elems
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
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
        let array = UArray::from_iter(SHAPE2X2, matrix);
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
}
