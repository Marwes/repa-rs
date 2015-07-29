use std::cmp::PartialEq;
use std::ops::Deref;
use std::iter::IntoIterator;

#[cfg(test)]
use quickcheck::{Arbitrary, Gen};

use shape::Shape;

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


pub struct Iter<S> {
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

pub fn iter<S>(s: &S) -> Iter<&S>
    where S: Source {
    range_iter(s, 0, s.extent().size())
}

pub fn range_iter<S>(s: &S, start: usize, end: usize) -> Iter<&S>
    where S: Source {
    assert!(start <= end);
    assert!(end <= s.extent().size());
    Iter { index: start, end: end, source: s }
}

#[derive(Clone, Debug)]
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

#[cfg(test)]
impl <S, E> Arbitrary for UArray<S, Vec<E>>
    where S: Arbitrary + Shape + Send + 'static
        , E: Arbitrary + Clone + Send + Sync +'static {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        //Take the size^(-rank) to make sure the shape does not get to large
        let shape = <S as Arbitrary>::arbitrary(g);
        let rank = shape.rank();
        let shape = shape.map(|x| (x as f64).powf(-(rank as f64)).ceil() as usize );
        let mut elems = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            elems.push(Arbitrary::arbitrary(g));
        }
        UArray { shape: shape, elems: elems }
    }

    fn shrink(&self) -> Box<Iterator<Item=UArray<S, Vec<E>>>> {
        use super::{extract, compute_s};
        let arr = self.clone();
        Box::new(self.extent().shrink().map(move |shape| compute_s(&extract(Shape::zero_dim(), shape, &arr))))
    }
}

pub trait Select<Args> {
    type Output;
    fn select(&self, args: Args) -> Self::Output;
}

pub struct SelectFn<F>(F);

impl <F, A, R> Select<(A,)> for SelectFn<F>
    where F: Fn(A) -> R {
    type Output = R;
    fn select(&self, args: (A,)) -> Self::Output {
        (self.0)(args.0)
    }
}

pub struct DArray<S, F>
    where S: Shape {
    shape: S,
    f: F
}

impl <S, F, E: Send + Sync> Source for DArray<S, F>
    where S: Shape
        , F: for<'a> Select<(&'a S,), Output=E> + Sync {
    type Element = E;
    type Shape = S;

    fn extent(&self) -> &<Self as Source>::Shape {
        &self.shape
    }
    fn index(&self, index: &<Self as Source>::Shape) -> E {
        if !self.extent().check_bounds(index) {
            panic!("Array out of bounds: {:?} {:?}", self.extent().size(), index.size())
        }
        self.f.select((index,))
    }
    fn linear_index(&self, index: usize) -> E {
        self.index(&self.shape.from_index(index))
    }
    unsafe fn unsafe_index(&self, index: &<Self as Source>::Shape) -> E {
        self.f.select((index,))
    }
    unsafe fn unsafe_linear_index(&self, index: usize) -> E {
        self.unsafe_index(&self.shape.from_index(index))
    }
}

impl <E, O, S, F> PartialEq<O> for DArray<S, F>
    where O: Source<Shape=S, Element=E>
        , S: Shape + PartialEq
        , F: for<'a> Select<(&'a S,), Output=E> + Sync
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

pub fn from_function<S, F, B>(shape: S, f: F) -> DArray<S, SelectFn<F>>
    where F: Fn(&S) -> B, S: Shape {
    DArray {
        shape: shape,
        f: SelectFn(f)
    }
}

pub fn from_select<S, F, B>(shape: S, f: F) -> DArray<S, F>
    where F: for<'a> Select<(&'a S,), Output=B>, S: Shape {
    DArray {
        shape: shape,
        f: f
    }
}
