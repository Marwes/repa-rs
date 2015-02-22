#![feature(core, unboxed_closures)]
use std::iter::IntoIterator;

#[derive(Clone)]
pub struct Z;

#[derive(Clone)]
pub struct Cons<T>(T, usize);

pub trait Shape: Clone {
    fn rank(&self) -> usize;
    fn zero_dim() -> Self;
    fn unit_dim() -> Self;
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


pub trait Source {
    type Element;
    type Sh: Shape;

    fn extent(&self) -> &<Self as Source>::Sh;
    fn index(&self, index: &<Self as Source>::Sh) -> <Self as Source>::Element;
    fn linear_index(&self, index: usize) -> <Self as Source>::Element;
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

impl <E: Copy, S: Shape> Source for UArray<S, E> {
    type Element = E;
    type Sh = S;

    fn extent(&self) -> &S {
        &self.shape
    }
    fn index(&self, index: &S) -> E {
        self.elems[self.shape.to_index(index)]
    }
    fn linear_index(&self, index: usize) -> E {
        self.elems[index]
    }
}

pub struct DArray<S, F>
    where S: Shape
        , F: for<'a> Fn<(&'a S,)> {
    shape: S,
    f: F
}

impl <S, F, E> Source for DArray<S, F>
    where S: Shape
        , F: for<'a> Fn(&'a S) -> E {
    type Element = E;
    type Sh = S;

    fn extent(&self) -> &<Self as Source>::Sh {
        &self.shape
    }
    fn index(&self, index: &<Self as Source>::Sh) -> E {
        (self.f)(index)
    }
    fn linear_index(&self, index: usize) -> E {
        (self.f)(&self.shape.from_index(index))
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

impl <'a, S, A, B, F> Fn<(&'a <S as Source>::Sh,)> for MapFn<S, F>
    where F: Fn(A) -> B
        , S: Source<Element=A> {
    type Output = B;
    extern "rust-call" fn call(&self, (sh,): (&<S as Source>::Sh,)) -> B {
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
}
