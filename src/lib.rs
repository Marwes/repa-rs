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

pub struct DArray<A, F> {
    array: A,
    f: F
}

impl <A, F, E> Source for DArray<A, F>
    where A: Source
        , F: Fn(<A as Source>::Element) -> E {
    type Element = E;
    type Sh = <A as Source>::Sh;

    fn extent(&self) -> &<Self as Source>::Sh {
        self.array.extent()
    }
    fn index(&self, index: &<Self as Source>::Sh) -> E {
        let e = self.array.index(index);
        (self.f)(e)
    }
    fn linear_index(&self, index: usize) -> E {
        let e = self.array.linear_index(index);
        (self.f)(e)
    }
}

pub fn map<S, F, B>(f: F, array: &S) -> DArray<&S, F>
    where F: Fn(<S as Source>::Element) -> B, S: Source {
    DArray {
        array: array,
        f: f
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

    #[test]
    fn index() {
        let shape = Cons(Cons(Z, 2), 2);
        let i_0_0 = Cons(Cons(Z, 0), 0);
        assert_eq!(shape.to_index(&i_0_0), 0);
    }

    #[test]
    fn array_index() {
        let d2 = Cons(Cons(Z, 2), 2);
        let matrix = vec![1, 2
                        , 3, 4];
        let array = UArray::from_iter(d2, matrix);
        assert_eq!(array.index(&Cons(Cons(Z, 0), 0)), 1);
        let delayed = map(|x| x * 2, &array);
        assert_eq!(delayed.index(&Cons(Cons(Z, 0), 0)), 2);
        assert_eq!(delayed.index(&Cons(Cons(Z, 1), 0)), 6);
    }
}
