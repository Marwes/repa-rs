use std::marker::PhantomData;

use shape::{Cons, Z, Shape};

pub struct All;
pub struct Any<S>(PhantomData<S>);

pub fn any<S>() -> Any<S> { Any(PhantomData) }

pub trait Slice {
    type Full: Shape;
    type Slice: Shape;
    fn slice_of_full(&self, full: Self::Full) -> Self::Slice;
    fn full_of_slice(&self, slice: Self::Slice) -> Self::Full;
}

impl Slice for Z {
    type Full = Z;
    type Slice = Z;
    fn slice_of_full(&self, _full: Z) -> Z { Z }
    fn full_of_slice(&self, _slice: Z) -> Z { Z }
}

impl <S> Slice for Any<S>
    where S: Shape {
    type Full = S;
    type Slice = S;
    fn slice_of_full(&self, full: S) -> S { full }
    fn full_of_slice(&self, slice: S) -> S { slice }
}

impl <S> Slice for Cons<S, All>
    where S: Slice {
    type Full = Cons<<S as Slice>::Full, usize>;
    type Slice = Cons<<S as Slice>::Slice, usize>;
    fn slice_of_full(&self, full: Cons<<S as Slice>::Full, usize>) -> Cons<<S as Slice>::Slice, usize> {
        Cons(self.0.slice_of_full(full.0), full.1)
    }
    fn full_of_slice(&self, slice: Cons<<S as Slice>::Slice, usize>) -> Cons<<S as Slice>::Full, usize> {
        Cons(self.0.full_of_slice(slice.0), slice.1)
    }
}

impl <S> Slice for Cons<S, usize>
    where S: Slice {
    type Full = Cons<<S as Slice>::Full, usize>;
    type Slice = <S as Slice>::Slice;
    fn slice_of_full(&self, full: Cons<<S as Slice>::Full, usize>) -> <S as Slice>::Slice {
        self.0.slice_of_full(full.0)
    }
    fn full_of_slice(&self, slice: <S as Slice>::Slice) -> Cons<<S as Slice>::Full, usize> {
        Cons(self.0.full_of_slice(slice), self.1)
    }
}
