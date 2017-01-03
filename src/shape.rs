use std::fmt;

#[cfg(test)]
use quickcheck::{Arbitrary, Gen};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Z;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cons<T, U = usize>(pub T, pub U);

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
    fn map<F>(&self, f: F) -> Self where F: FnMut(usize) -> usize;
}
impl Shape for Z {
    #[inline]
    fn rank(&self) -> usize {
        0
    }
    #[inline]
    fn zero_dim() -> Z {
        Z
    }
    #[inline]
    fn unit_dim() -> Z {
        Z
    }
    #[inline]
    fn add_dim(&self, _: &Z) -> Z {
        Z
    }
    #[inline]
    fn intersect_dim(&self, _: &Z) -> Z {
        Z
    }
    #[inline]
    fn size(&self) -> usize {
        1
    }
    #[inline]
    fn to_index(&self, _: &Z) -> usize {
        0
    }
    #[inline]
    fn from_index(&self, _: usize) -> Z {
        Z
    }
    #[inline]
    fn check_bounds(&self, _index: &Z) -> bool {
        true
    }
    #[inline]
    fn map<F>(&self, _: F) -> Z
        where F: FnMut(usize) -> usize
    {
        Z
    }
}
impl<T: Shape> Shape for Cons<T> {
    #[inline]
    fn rank(&self) -> usize {
        self.0.rank() + 1
    }
    #[inline]
    fn zero_dim() -> Cons<T> {
        Cons(Shape::zero_dim(), 0)
    }
    #[inline]
    fn unit_dim() -> Cons<T> {
        Cons(Shape::unit_dim(), 1)
    }
    #[inline]
    fn add_dim(&self, other: &Cons<T>) -> Cons<T> {
        Cons(self.0.add_dim(&other.0), self.1 + other.1)
    }
    #[inline]
    fn intersect_dim(&self, other: &Cons<T>) -> Cons<T> {
        Cons(self.0.intersect_dim(&other.0),
             ::std::cmp::min(self.1, other.1))
    }
    #[inline]
    fn size(&self) -> usize {
        self.1 * self.0.size()
    }
    #[inline]
    fn to_index(&self, index: &Cons<T>) -> usize {
        self.0.to_index(&index.0) * self.1 + index.1
    }
    #[inline]
    fn from_index(&self, i: usize) -> Cons<T> {
        let r = if self.0.rank() == 0 { i } else { i % self.1 };
        Cons(self.0.from_index(i / self.1), r)
    }
    #[inline]
    fn check_bounds(&self, index: &Cons<T>) -> bool {
        index.1 < self.1 && self.0.check_bounds(&index.0)
    }
    #[inline]
    fn map<F>(&self, mut f: F) -> Cons<T>
        where F: FnMut(usize) -> usize
    {
        let i = f(self.1);
        Cons(self.0.map(f), i)
    }
}

impl fmt::Display for Z {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Z")
    }
}

impl<S> fmt::Display for Cons<S>
    where S: fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} :. {}", self.0, self.1)
    }
}

#[cfg(test)]
impl Arbitrary for Z {
    fn arbitrary<G: Gen>(_: &mut G) -> Self {
        Z
    }
}
#[cfg(test)]
impl<T, U> Arbitrary for Cons<T, U>
    where T: Arbitrary + Send + 'static,
          U: Arbitrary + Send + 'static
{
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        Cons(Arbitrary::arbitrary(g), Arbitrary::arbitrary(g))
    }

    fn shrink(&self) -> Box<Iterator<Item = Self>> {
        let Cons(x, y) = self.clone();
        Box::new(self.0
            .shrink()
            .map(move |x| Cons(x, y.clone()))
            .chain(self.1.shrink().map(move |y| Cons(x.clone(), y))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;

    #[quickcheck]
    fn to_index_from_index(s: (usize, usize, usize), i: (usize, usize, usize)) -> TestResult {
        let shape = Cons(Cons(Cons(Z, s.0), s.1), s.2);
        let index = Cons(Cons(Cons(Z, i.0), i.1), i.2);
        if !shape.check_bounds(&index) {
            return TestResult::discard();
        }
        TestResult::from_bool(shape.from_index(shape.to_index(&index)) == index)
    }
}
