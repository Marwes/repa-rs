#![cfg_attr(test, feature(plugin, custom_attribute))]
#![cfg_attr(test, plugin(quickcheck_macros))]
use std::marker::PhantomData;
use std::default::Default;
use std::mem;
use std::thread;

use shape::{Cons, Shape};
use source::{from_select, iter, range_iter, Source, Select, DArray, UArray};
use slice::Slice;

#[cfg(test)]
extern crate quickcheck;

pub mod shape;
pub mod slice;
pub mod source;

pub struct MapFn<S, F> {
    source: S,
    f: F
}

impl <'a, S, A, B, F> Select<(&'a <S as Source>::Shape,)> for MapFn<S, F>
    where A: Send + Sync
        , F: Fn(A) -> B
        , S: Source<Element=A> {
    type Output = B;
    fn select(&self, (sh,): (&<S as Source>::Shape,)) -> B {
        let e = unsafe { self.source.unsafe_index(sh) };
        (self.f)(e)
    }
}

pub fn map<S, F, B>(f: F, array: S) -> DArray<<S as Source>::Shape, MapFn<S, F>>
    where F: Fn(<S as Source>::Element) -> B, S: Source {
    from_select(array.extent().clone(), MapFn { source: array, f: f })
}

pub struct ExtractFn<S>
    where S: Source {
    source: S,
    start: <S as Source>::Shape,
}

impl <'a, S> Select<(&'a <S as Source>::Shape,)> for ExtractFn<S>
    where S: Source {
    type Output = <S as Source>::Element;
    fn select(&self, (sh,): (&<S as Source>::Shape,)) -> <S as Source>::Element {
        let i = self.start.add_dim(sh);
        unsafe { self.source.unsafe_index(&i) }
    }
}

pub fn extract<S>(start: <S as Source>::Shape, size: <S as Source>::Shape, array: S) -> DArray<<S as Source>::Shape, ExtractFn<S>>
    where S: Source {
    if !array.extent().map(|i| i + 1).check_bounds(&start.add_dim(&size)) {
        panic!("extract: out of bounds")
    }
    from_select(size, ExtractFn { source: array, start: start })
}

pub struct TransposeFn<S>
    where S: Source {
    source: S
}

impl <'a, S, Sh> Select<(&'a <S as Source>::Shape,)> for TransposeFn<S>
    where S: Source<Shape=Cons<Cons<Sh>>>
        , Sh: Shape {
    type Output = <S as Source>::Element;
    fn select(&self, (sh,): (&<S as Source>::Shape,)) -> <S as Source>::Element {
        let &Cons(Cons(ref rest, x), y) = sh;
        unsafe { self.source.unsafe_index(&Cons(Cons(rest.clone(), y), x)) }
    }
}

pub fn transpose<S, Sh>(array: S) -> DArray<<S as Source>::Shape, TransposeFn<S>>
    where S: Source<Shape=Cons<Cons<Sh>>>
        , Sh: Shape {
    let Cons(Cons(rest, x), y) = array.extent().clone();
    from_select(Cons(Cons(rest, y), x), TransposeFn { source: array })
}

pub struct ZipWithFn<S1, S2, F>
    where S1: Source
        , S2: Source {
    lhs: S1,
    rhs: S2,
    f: F
}

impl <'a, S1, S2, Sh, F, O> Select<(&'a <S1 as Source>::Shape,)> for ZipWithFn<S1, S2, F>
    where Sh: Shape
        , S1: Source<Shape=Sh>
        , S2: Source<Shape=Sh>
        , F: Fn(<S1 as Source>::Element, <S2 as Source>::Element) -> O {
    type Output = O;
    fn select(&self, (sh,): (&Sh,)) -> O {
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
    from_select(lhs.extent().intersect_dim(rhs.extent()), ZipWithFn { lhs: lhs, rhs: rhs, f: f })
}

pub struct TraverseFn<S, T>
    where S: Source {
    source: S,
    transform: T
}

impl <'a, S, Sh, T, B> Select<(&'a Sh,)> for TraverseFn<S, T>
    where Sh: Shape
        , S: Source
        , T: for<'b, 'c> Fn(&'b S, &'c Sh) -> B {
    type Output = B;
    fn select(&self, (sh,): (&Sh,)) -> B {
        (self.transform)(&self.source, sh)
    }
}

pub fn traverse<S, Sh, F, T, A, B>(array: S, new_shape: F, transform: T) -> DArray<Sh, TraverseFn<S, T>>
    where S: Source
        , Sh: Shape
        , F: FnOnce(&<S as Source>::Shape) -> Sh
        , T: Fn(&S, &Sh) -> B {

    let shape = new_shape(array.extent());
    from_select(shape, TraverseFn { source: array, transform: transform })
}

pub struct SliceFn<A, S>
    where A: Source
        , S: Slice {
    array: A,
    slice: S
}

impl <'a, A, S> Select<(&'a <S as Slice>::Slice,)> for SliceFn<A, S>
    where S: Slice
        , A: Source<Shape=<S as Slice>::Full> {
    type Output = <A as Source>::Element;
    fn select(&self, (sh,): (&<S as Slice>::Slice,)) -> <A as Source>::Element {
        //Get the index in the full array
        let full_index = self.slice.full_of_slice(sh.clone());
        self.array.index(&full_index)
    }
}

pub fn slice<A, S>(array: A, slice: S) -> DArray<<S as Slice>::Slice, SliceFn<A, S>>
    where A: Source<Shape=<S as Slice>::Full>
        , S: Slice {
    from_select(slice.slice_of_full(array.extent().clone()), SliceFn { array: array, slice: slice })
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
    UArray::new(shape.clone(), elems)
}

pub fn fold_p<F, S, Sh>(ref f: F, ref e: <S as Source>::Element, array: &S) -> UArray<Sh, Vec<<S as Source>::Element>>
    where Sh: Shape
        , S: Source<Shape=Cons<Sh>>
        , F: Fn(<S as Source>::Element, <S as Source>::Element) -> <S as Source>::Element + Sync
        , <S as Source>::Element: Clone {
    let &Cons(ref shape, size) = array.extent();
    let new_size = shape.size();
    let elems = parallel_vec(new_size, |index| {
        let offset = index * size;
        let mut r = e.clone();
        for e in range_iter(array, offset, (offset + size)) {
            r = f(r, e);
        }
        r
    });
    UArray::new(shape.clone(), elems)
}

pub fn compute_s<S>(array: &S) -> UArray<<S as Source>::Shape, Vec<<S as Source>::Element>>
    where S: Source
        , <S as Source>::Element: Clone {
    UArray::new(array.extent().clone(), iter(array).collect())
}

pub fn compute_p<S>(array: &S) -> UArray<<S as Source>::Shape, Vec<<S as Source>::Element>>
    where S: Source
        , <S as Source>::Element: Default + Clone {
    let extent = array.extent();
    let size = extent.size();
    let elems = parallel_vec(size, |i| unsafe { array.unsafe_linear_index(i) });
    UArray::new(extent.clone(), elems)
}

struct ParallelChunk<'a, T: 'a> {
    slice: *mut T,
    len: usize,
    fill_count: usize,
    _marker: PhantomData<&'a mut T>
}

unsafe impl <'a, T> Send for ParallelChunk<'a, T> where &'a mut [T]: Send { }

impl <'a, T> Drop for ParallelChunk<'a, T> {
    fn drop(&mut self) {
        use std::ptr;
        if self.slice != ptr::null_mut() {
            for i in 0..self.fill_count {
                unsafe { ptr::read(self.slice.offset(i as isize)); }
            }
        }
    }
}

impl <'a, T> ParallelChunk<'a, T> {

    fn len(&self) -> usize { self.len }

    fn push(&mut self, value: T) {
        use std::ptr;
        if self.fill_count >= self.len {
            panic!("push called to full guard");
        }
        unsafe { ptr::write(self.slice.offset(self.fill_count as isize), value) };
        self.fill_count += 1;
    }
}

///A wrapper around a vector which has a capacity for atleast `size` elements
struct ParallelVec<'a, T: 'a> {
    vec: &'a mut Vec<T>,
    chunk_len: usize,
    guards: Vec<ParallelChunk<'a, T>>
}

impl <'a, T> Drop for ParallelVec<'a, T> {
    fn drop(&mut self) {
        self.guards.reverse();
        while let Some(guard) = self.guards.pop() {
            //As long as each chunk was filled completely we we know that the vector has been
            //filled continously
            if guard.len == guard.fill_count {
                let new_len = self.vec.len() + guard.len;
                unsafe { self.vec.set_len(new_len); }
            }
            else {
                //This chunk was not completely filled so we just drop the remaining chunks
                break
            }
        }
    }
}

impl <'a, T> ParallelVec<'a, T> {
    fn new(vec: &'a mut Vec<T>, size: usize, chunks: usize) -> ParallelVec<'a, T> {
        //Make sure the vec is empty so that existing elements does not get overwritten
        //when an element is pushed to a chunk
        vec.clear();
        let additional = size - vec.len();
        vec.reserve(additional);
        let mut guards = Vec::new();

        let slice_len = (size + chunks - 1) / chunks;
        let mut ptr = vec.as_mut_ptr();
        let end = unsafe { ptr.offset(size as isize) };
        while ptr < end {
            let chunk_len = ::std::cmp::min(slice_len, (end as usize - ptr as usize) / ::std::mem::size_of::<T>());
            guards.push(ParallelChunk { slice: ptr, len: chunk_len, fill_count: 0, _marker: PhantomData });
            unsafe { ptr = ptr.offset(slice_len as isize) }
        }
        ParallelVec { vec: vec, chunk_len: slice_len, guards: guards }
    }
    
    fn chunks(&mut self) -> (usize, &mut [ParallelChunk<'a, T>]) {
        (self.chunk_len, &mut self.guards)
    }
}

/// Spawns a scoped thread which must be joined before the lifetime of `f`
/// goes out of scope
unsafe fn spawn_scoped<F>(f: F) -> thread::JoinHandle<()>
    where F: FnOnce() + Send {
    let box_f = Box::new(f);
    let p = &*box_f as *const F as usize;
    mem::forget(box_f);
    thread::spawn(move || {
        let box_f: Box<F> = mem::transmute(p);
        box_f()
    })
}

fn parallel_vec<E, F>(len: usize, ref f: F) -> Vec<E>
    where E: Send + Sync
        , F: Fn(usize) -> E + Sync {
    let mut elems = Vec::with_capacity(len);
    let slices = 4;//TODO
    unsafe {
        let mut parallel = ParallelVec::new(&mut elems, len, slices);
        let mut guards = Vec::new();
        let (slice_len, chunks) = parallel.chunks();
        for (i, chunk) in chunks.iter_mut().enumerate() {
            let offset = i * slice_len;
            let x = spawn_scoped(move || {
                for j in offset..(offset + chunk.len()) {
                    chunk.push(f(j));
                }
            });
            guards.push(x);
        }
        for guard in guards {
            guard.join().unwrap();
        }
    }
    elems
}

#[cfg(test)]
mod tests {
    use super::*;
    use shape::{Cons, Z, Shape};
    use source::{from_function, from_select, iter, Source, Select, DArray, UArray};
    use slice::{All, any};
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

    #[test]
    fn slice_test() {
        let m = from_function(SHAPE2X2, |i| SHAPE2X2.to_index(i));
        let column0 = slice(&m, Cons(any(), 0));
        assert_eq!(column0.index(&Cons(Z, 0)), 0);
        assert_eq!(column0.index(&Cons(Z, 1)), 2);
        let row1 = slice(&m, Cons(Cons(any(), 1), All));
        assert_eq!(row1.index(&Cons(Z, 0)), 2);
        assert_eq!(row1.index(&Cons(Z, 1)), 3);
    }

    pub struct Multiply<A, B> {
        lhs: A,
        rhs: B
    }

    impl <'a, A, B> Select<(&'a <A as Source>::Shape,)> for Multiply<A, B>
        where A: Source<Shape=Cons<Cons<Z>>, Element=i64>
            , B: Source<Shape=Cons<Cons<Z>>, Element=A::Element> {
        type Output = B::Element;
        fn select(&self, (sh,): (&<A as Source>::Shape,)) -> B::Element {
            let Cons(Cons(_, i), j) = *sh;
            let row = slice(&self.lhs, Cons(Cons(any(), i), All));
            let column = slice(&self.rhs, Cons(any(), j));
            iter(&zip_with(&row, &column, |l, r| l * r))
                .fold(0, |l, r| l + r)
        }
    }

    fn multiply<A, B>(lhs: A, rhs: B) -> DArray<Cons<Cons<Z>>, Multiply<A, B>>
        where A: Source<Shape=Cons<Cons<Z>>, Element=i64>
            , B: Source<Shape=Cons<Cons<Z>>, Element=A::Element> {
        let Cons(Cons(_, h), _) = *lhs.extent();
        let Cons(Cons(_, _), w) = *rhs.extent();
        from_select(Cons(Cons(Z, h), w), Multiply { lhs: lhs, rhs: rhs })
    }

    type Mat2D = UArray<Cons<Cons<Z>>, Vec<i64>>;
    #[quickcheck]
    fn matrix(a: Mat2D, b: Mat2D) -> TestResult {
        let &Cons(Cons(_, ah), w) = a.extent();
        let &Cons(Cons(_, h), bw) = b.extent();
        if w != h || ah == 0 || bw == 0 {
            return TestResult::discard();
        }
        TestResult::from_bool(transpose(multiply(&a, &b)) == multiply(transpose(&b), transpose(&a)))
    }
    
    #[test]
    #[should_panic]
    fn slice_out_of_bounds() {
        let m = from_function(SHAPE2X2, |i| SHAPE2X2.to_index(i));
        let column0 = slice(&m, Cons(any(), 0));
        column0.index(&Cons(Z, 2));
    }

    #[quickcheck]
    fn transpose_test(vs: Vec<i32>, size: usize) -> bool {
        //size is the length of one side of the 2 dimensional matrix
        let size = if vs.len() / 2 == 0 { 0 } else { size % (vs.len() / 2) };
        let m = UArray::new(Cons(Cons(Z, size), vs.len() / 2), vs);
        transpose(transpose(&m)) == m
    }
}
