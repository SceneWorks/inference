//! CPU frozen `torch.topk` partition order (libc++); shared by all backends.
//! nth partition is adapted from LLVM libc++ (Apache-2.0 WITH LLVM-exception).
//! Sorting the first k-1 entries is immaterial to a block mask: Bresenham
//! keeps either all k entries or precisely those first k-1 entries.
fn before(a: (f32, usize), b: (f32, usize)) -> bool {
    a.0 > b.0
}
fn sort3(v: &mut [(f32, usize)], a: usize, b: usize, c: usize) -> usize {
    let mut swaps = 0;
    if before(v[b], v[a]) {
        v.swap(a, b);
        swaps += 1;
    }
    if before(v[c], v[b]) {
        v.swap(b, c);
        swaps += 1;
        if before(v[b], v[a]) {
            v.swap(a, b);
            swaps += 1;
        }
    }
    swaps
}
fn nth(v: &mut [(f32, usize)], kth: usize) {
    let mut first = 0;
    let mut last = v.len();
    loop {
        let len = last - first;
        if len <= 1 {
            return;
        }
        if len == 2 {
            if before(v[last - 1], v[first]) {
                v.swap(first, last - 1);
            }
            return;
        }
        if len == 3 {
            sort3(v, first, first + 1, last - 1);
            return;
        }
        if len <= 7 {
            for a in first..last - 1 {
                let mut m = a;
                for b in a + 1..last {
                    if before(v[b], v[m]) {
                        m = b;
                    }
                }
                if m != a {
                    v.swap(a, m);
                }
            }
            return;
        }
        let mut m = first + len / 2;
        let mut swaps = sort3(v, first, m, last - 1);
        let mut i = first;
        let mut j = last - 1;
        if !before(v[i], v[m]) {
            let found = loop {
                j -= 1;
                if i == j {
                    break false;
                }
                if before(v[j], v[m]) {
                    break true;
                }
            };
            if found {
                v.swap(i, j);
                swaps += 1;
            } else {
                i += 1;
                j = last - 1;
                if !before(v[first], v[j]) {
                    loop {
                        if i == j {
                            return;
                        }
                        if before(v[first], v[i]) {
                            v.swap(i, j);
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
                if i == j {
                    return;
                }
                loop {
                    while !before(v[first], v[i]) {
                        i += 1;
                    }
                    loop {
                        j -= 1;
                        if !before(v[first], v[j]) {
                            break;
                        }
                    }
                    if i >= j {
                        break;
                    }
                    v.swap(i, j);
                    i += 1;
                }
                if kth < i {
                    return;
                }
                first = i;
                continue;
            }
        }
        i += 1;
        if i < j {
            loop {
                while before(v[i], v[m]) {
                    i += 1;
                }
                loop {
                    j -= 1;
                    if before(v[j], v[m]) {
                        break;
                    }
                }
                if i >= j {
                    break;
                }
                v.swap(i, j);
                swaps += 1;
                if m == i {
                    m = j;
                }
                i += 1;
            }
        }
        if i != m && before(v[m], v[i]) {
            v.swap(i, m);
            swaps += 1;
        }
        if kth == i {
            return;
        }
        if swaps == 0 {
            let (a, b) = if kth < i { (first, i) } else { (i, last) };
            if (a + 1..b).all(|p| !before(v[p], v[p - 1])) {
                return;
            }
        }
        if kth < i {
            last = i;
        } else {
            first = i + 1;
        }
    }
}
fn sift(v: &mut [(f32, usize)], mut root: usize, n: usize) {
    // libc++'s heap is inverse to the descending comparison: the least
    // wanted value is at the root, and the right child wins equivalent ties.
    let value = v[root];
    loop {
        let mut child = 2 * root + 1;
        if child >= n {
            break;
        }
        if child + 1 < n && before(v[child], v[child + 1]) {
            child += 1;
        }
        if before(v[child], value) {
            break;
        }
        v[root] = v[child];
        root = child;
    }
    v[root] = value;
}
pub(super) fn topk(mut v: Vec<(f32, usize)>, k: usize) -> Vec<usize> {
    if k * 64 <= v.len() {
        for root in (0..k / 2).rev() {
            sift(&mut v, root, k);
        }
        for i in k..v.len() {
            if before(v[i], v[0]) {
                v.swap(i, 0);
                sift(&mut v, 0, k);
            }
        }
        for n in (2..=k).rev() {
            v.swap(0, n - 1);
            sift(&mut v, 0, n - 1);
        }
    } else {
        nth(&mut v, k - 1);
    }
    v[..k].iter().map(|p| p.1).collect()
}
