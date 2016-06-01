"""LInked List sparse matrix class

These functions and classes are modified versions of the SciPy sparse matrix
utilities. The code comes under the following license:

Copyright (c) 2001, 2002 Enthought, Inc.
All rights reserved.

Copyright (c) 2003-2012 SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Enthought nor the names of the SciPy Developers
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import division, print_function
from bisect import bisect_left
import numpy as np


def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    """

    int32max = np.iinfo(np.int32).max

    dtype = np.intc
    if maxval is not None:
        if maxval > int32max:
            dtype = np.int64

    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)

    for arr in arrays:
        arr = np.asarray(arr)
        if arr.dtype > np.int32:
            if check_contents:
                if arr.size == 0:
                    # a bigger type not needed
                    continue
                elif np.issubdtype(arr.dtype, np.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= np.iinfo(np.int32).min and \
                            maxval <= np.iinfo(np.int32).max:
                        # a bigger type not needed
                        continue

            dtype = np.int64
            break

    return dtype


def _slicetoarange(j, shape):
    """ Given a slice object, use numpy arange to change it to a 1D
    array.
    """
    start, stop, step = j.indices(shape)
    return np.arange(start, stop, step)


def _check_ellipsis(index):
    """Process indices with Ellipsis. Returns modified index."""
    if index is Ellipsis:
        return (slice(None), slice(None))
    elif isinstance(index, tuple):
        # Find first ellipsis
        for j, v in enumerate(index):
            if v is Ellipsis:
                first_ellipsis = j
                break
        else:
            first_ellipsis = None

        # Expand the first one
        if first_ellipsis is not None:
            # Shortcuts
            if len(index) == 1:
                return (slice(None), slice(None))
            elif len(index) == 2:
                if first_ellipsis == 0:
                    if index[1] is Ellipsis:
                        return (slice(None), slice(None))
                    else:
                        return (slice(None), index[1])
                else:
                    return (index[0], slice(None))

            # General case
            tail = ()
            for v in index[first_ellipsis+1:]:
                if v is not Ellipsis:
                    tail = tail + (v,)
            nd = first_ellipsis + len(tail)
            nslice = max(0, 2 - nd)
            return index[:first_ellipsis] + (slice(None),)*nslice + tail

    return index


def _boolean_index_to_array(i):
    if i.ndim > 1:
        raise IndexError('invalid index shape')
    return i.nonzero()[0]


class IndexMixin(object):
    """
    This class simply exists to hold the methods necessary for fancy indexing.
    """

    def _unpack_index(self, index):
        """ Parse index. Always return a tuple of the form (row, col).
        Where row/col is a integer, slice, or array of integers.
        """
        if (isinstance(index, (spmatrix, np.ndarray)) and
                index.ndim == 2 and index.dtype.kind == 'b'):
            return index.nonzero()

        # Parse any ellipses.
        index = _check_ellipsis(index)

        # Next, parse the tuple or object
        if isinstance(index, tuple):
            if len(index) == 2:
                row, col = index
            elif len(index) == 1:
                row, col = index[0], slice(None)
            else:
                raise IndexError('invalid number of indices')
        else:
            row, col = index, slice(None)

        # Next, check for validity, or transform the index as needed.
        row, col = self._check_boolean(row, col)
        return row, col

    def _check_boolean(self, row, col):
        # Supporting sparse boolean indexing with both row and col does
        # not work because spmatrix.ndim is always 2.
        if isspmatrix(row) or isspmatrix(col):
            raise IndexError("Indexing with sparse matrices is not supported"
                             " except boolean indexing where matrix and index"
                             " are equal shapes.")
        if isinstance(row, np.ndarray) and row.dtype.kind == 'b':
            row = self._boolean_index_to_array(row)
        if isinstance(col, np.ndarray) and col.dtype.kind == 'b':
            col = self._boolean_index_to_array(col)
        return row, col

    def _index_to_arrays(self, i, j):
        i, j = self._check_boolean(i, j)
        i_slice = isinstance(i, slice)
        if i_slice:
            i = _slicetoarange(i, self.shape[0])[:, None]
        else:
            i = np.atleast_1d(i)
        if isinstance(j, slice):
            j = _slicetoarange(j, self.shape[1])[None, :]
            if i.ndim == 1:
                i = i[:, None]
            elif not i_slice:
                raise IndexError('index returns 3-dim structure')
        elif isscalarlike(j):
            # row vector special case
            j = np.atleast_1d(j)
            if i.ndim == 1:
                i, j = np.broadcast_arrays(i, j)
                i = i[:, None]
                j = j[:, None]
                return i, j
        else:
            j = np.atleast_1d(j)
            if i_slice and j.ndim > 1:
                raise IndexError('index returns 3-dim structure')
        return i, j


def issequence(t):
    return (isinstance(t, (list, tuple)) and
            (len(t) == 0 or np.isscalar(t[0]))) or \
           (isinstance(t, np.ndarray) and (t.ndim == 1))


def isdense(x):
    return isinstance(x, np.ndarray)


def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


def isintlike(x):
    """Is x appropriate as an index into a sparse matrix? Returns True
    if it can be cast safely to a machine int.
    """
    if issequence(x):
        return False
    else:
        try:
            if int(x) == x:
                return True
            else:
                return False
        except TypeError:
            return False


def isshape(x):
    """Is x a valid 2-tuple of dimensions?
    """
    try:
        # Assume it's a tuple of matrix dimensions (M, N)
        (M, N) = x
    except TypeError:
        return False
    else:
        if isintlike(M) and isintlike(N):
            if np.ndim(M) == 0 and np.ndim(N) == 0:
                return True
        return False


def getdtype(dtype, a=None, default=None):
    """Function used to simplify argument processing.  If 'dtype' is not
    specified (is None), returns a.dtype; otherwise returns a np.dtype
    object created from the specified dtype argument.  If 'dtype' and 'a'
    are both None, construct a data type out of the 'default' parameter.
    Furthermore, 'dtype' must be in 'allowed' set.
    """
    if dtype is None:
        try:
            newdtype = a.dtype
        except AttributeError:
            if default is not None:
                newdtype = np.dtype(default)
            else:
                raise TypeError("could not interpret data type")
    else:
        newdtype = np.dtype(dtype)

    return newdtype


def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    return (np.array([0], dtype=dtype) * scalar).dtype


class spmatrix(object):
    """ This class provides a base class for all sparse matrices.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    """

    __array_priority__ = 10.1
    ndim = 2

    def __init__(self):
        self.format = self.__class__.__name__[:3]
        self._shape = None
        if self.format == 'spm':
            raise ValueError("This class is not intended"
                             "to be instantiated directly.")

    def get_shape(self):
        return self._shape


def lil_get1(M, N, rows, datas, i, j):
    """
    Get a single item from LIL matrix.

    Doesn't do output type conversion. Checks for bounds errors.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get

    Returns
    -------
    x
        Value at indices.

    """

    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    row = rows[i]
    data = datas[i]
    pos = bisect_left(row, j)

    if pos != len(data) and row[pos] == j:
        return data[pos]
    else:
        return 0


def _lil_fancy_get(M, N, rows, datas, new_rows, new_datas, i_idx, j_idx):
    for x in range(i_idx.shape[0]):
        new_row = []
        new_data = []

        for y in range(j_idx.shape[1]):
            i = i_idx[x, 0]
            j = j_idx[0, y]
            value = lil_get1(M, N, rows, datas, i, j)

            if value is not 0:
                # Object identity as shortcut
                new_row.append(y)
                new_data.append(value)

        new_rows[x] = new_row
        new_datas[x] = new_data


def lil_deleteat_nocheck(row, data, j):
    """
    Delete a single item from a row in LIL matrix.

    Doesn't check for bounds errors.

    Parameters
    ----------
    row, data
        Row data for LIL matrix.
    j : int
        Column index to delete at

    """
    pos = bisect_left(row, j)
    if pos < len(row) and row[pos] == j:
        del row[pos]
        del data[pos]


def lil_insertat_nocheck(row, data, j, x):
    """
    Insert a single item to LIL matrix.

    Doesn't check for bounds errors. Doesn't check for zero x.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """

    pos = bisect_left(row, j)
    if pos == len(row):
        row.append(j)
        data.append(x)
    elif row[pos] != j:
        row.insert(pos, j)
        data.insert(pos, x)
    else:
        data[pos] = x


def _lil_insert(M, N, rows, datas, i, j, x):
    """
    Insert a single item to LIL matrix.

    Checks for bounds errors and deletes item if x is zero.

    Parameters
    ----------
    M, N, rows, datas
        Shape and data arrays for a LIL matrix
    i, j : int
        Indices at which to get
    x
        Value to insert.

    """
    if i < -M or i >= M:
        raise IndexError('row index (%d) out of bounds' % (i,))
    if i < 0:
        i += M

    if j < -N or j >= N:
        raise IndexError('column index (%d) out of bounds' % (j,))
    if j < 0:
        j += N

    if x == 0:
        lil_deleteat_nocheck(rows[i], datas[i], j)
    else:
        lil_insertat_nocheck(rows[i], datas[i], j, x)


class lil_matrix(spmatrix, IndexMixin):
    """Row-based linked list sparse matrix

    This is an efficient structure for constructing sparse
    matrices incrementally.

    This can be instantiated in several ways:
        lil_matrix(D)
            with a dense matrix or rank-2 ndarray D

        lil_matrix(S)
            with another sparse matrix S (equivalent to S.tolil())

        lil_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of nonzero elements
    data
        LIL format data array of the matrix
    rows
        LIL format row index array of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the matrix sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse matrices
        - once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large matrices

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.


    """

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        spmatrix.__init__(self)
        self.dtype = getdtype(dtype, arg1, default=float)

        # First get the shape
        if isspmatrix(arg1):
            if isspmatrix_lil(arg1) and copy:
                A = arg1.copy()
            else:
                A = arg1.tolil()

            if dtype is not None:
                A = A.astype(dtype)

            self.shape = A.shape
            self.dtype = A.dtype
            self.rows = A.rows
            self.data = A.data
        elif isinstance(arg1, tuple):
            if isshape(arg1):
                if shape is not None:
                    raise ValueError('invalid use of shape parameter')
                M, N = arg1
                self.shape = (M, N)
                pre_rows = []
                pre_data = []
                for _ in range(M):
                    pre_rows.append([])
                    pre_data.append([])
                self.rows = pre_rows
                self.data = pre_data

            else:
                raise TypeError('unrecognized lil_matrix constructor usage')
        else:
            # assume A is dense
            try:
                A = np.asmatrix(arg1)
            except TypeError:
                raise TypeError('unsupported matrix type')

    def set_shape(self, shape):
        shape = tuple(shape)

        if len(shape) != 2:
            raise ValueError("Only two-dimensional sparse arrays "
                             "are supported.")
        try:
            shape = int(shape[0]), int(shape[1])  # floats, other weirdness
        except:
            raise TypeError('invalid shape')

        if not (shape[0] >= 0 and shape[1] >= 0):
            raise ValueError('invalid shape')

        if (self._shape != shape) and (self._shape is not None):
            try:
                self = self.reshape(shape)
            except NotImplementedError:
                raise NotImplementedError("Reshaping not implemented for %s." %
                                          self.__class__.__name__)
        self._shape = shape

    shape = property(fget=spmatrix.get_shape, fset=set_shape)

    def __iadd__(self, other):
        if isinstance(other, lil_matrix):
            if self.shape != other.shape:
                raise Exception("Shapes don't match!")
            # Super-retarded solution
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    a = self[i, j]
                    b = other[i, j]
                    if a != 0 or b != 0:
                        self[i, j] = a + b
            return self
        else:
            raise NotImplementedError

    def __isub__(self, other):
        self[:, :] = self - other
        return self

    def __imul__(self, other):
        if isscalarlike(other):
            self[:, :] = self * other
            return self
        else:
            raise NotImplementedError

    def __itruediv__(self, other):
        if isscalarlike(other):
            self[:, :] = self / other
            return self
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isscalarlike(other):
            return self._mul_scalar(other)
        else:
            raise NotImplementedError

    __rmul__ = __mul__

    # Whenever the dimensions change, empty lists should be created for each
    # row

    def getnnz(self, axis=None):
        """Get the count of explicitly-stored values (nonzeros)

        Parameters
        ----------
        axis : None, 0, or 1
            Select between the number of values across the whole matrix, in
            each column, or in each row.
        """
        if axis is None:
            return sum([len(rowvals) for rowvals in self.data])
        if axis < 0:
            axis += 2
        if axis == 0:
            out = np.zeros(self.shape[1])
            for row in self.rows:
                out[row] += 1
            return out
        elif axis == 1:
            return np.array([len(rowvals) for rowvals in self.data])
        else:
            raise ValueError('axis out of bounds')
    nnz = property(fget=getnnz)

    def __str__(self):
        val = ''
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                val += "  %s\t%s\n" % (str((i, j)), str(self.data[i][pos]))
        return val[:-1]

    def getcol(self, i):
        new = lil_matrix((self.shape[0], 1), dtype=self.dtype)
        for row_index, row in enumerate(self.rows):
            for column_index, column in enumerate(row):
                if column > i:
                    break
                if column == i:
                    new[row_index, 0] = self.data[row_index][column_index]
        return new

    def getrowview(self, i):
        """Returns a view of the 'i'th row (without copying).
        """
        new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i]
        new.data[0] = self.data[i]
        return new

    def getrow(self, i):
        """Returns a copy of the 'i'th row.
        """
        new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i][:]
        new.data[0] = self.data[i][:]
        return new

    def __getitem__(self, index):
        """Return the element(s) index=(i, j), where j may be a slice.
        This always returns a copy for consistency, since slices into
        Python lists return copies.
        """
        # Utilities found in IndexMixin
        if isinstance(index, int):
            return self.getrow(index)
        if isinstance(index[0], int) and isinstance(index[1], int):
            return lil_get1(self.shape[0], self.shape[1], self.rows, self.data,
                            index[0], index[1])
        i, j = self._unpack_index(index)
        i, j = self._index_to_arrays(i, j)
        new = lil_matrix((i.shape[0], j.shape[1]), dtype=self.dtype)
        _lil_fancy_get(self.shape[0], self.shape[1],
                       self.rows, self.data,
                       new.rows, new.data,
                       i, j)
        return new

    def __setitem__(self, index, x):
        # Scalar fast path first
        if isinstance(index, tuple) and len(index) == 2:
            i, j = index
            # Use isinstance checks for common index types; this is
            # ~25-50% faster than isscalarlike. Scalar index
            # assignment for other types is handled below together
            # with fancy indexing.
            if ((isinstance(i, int) or isinstance(i, np.integer)) and
                    (isinstance(j, int) or isinstance(j, np.integer))):
                x = self.dtype.type(x)
                if x.size > 1:
                    # Triggered if input was an ndarray
                    raise ValueError("Trying to assign a sequence to an item")
                _lil_insert(self.shape[0], self.shape[1],
                            self.rows, self.data,
                            i, j, x)
                return
        elif (isinstance(index, int) or isinstance(index, np.integer)) and \
                isinstance(x, lil_matrix) and x.shape == (1, self.shape[1]):
            self.rows[index] = x.rows[0][:]
            self.data[index] = x.data[0][:]
            return
        else:
            raise NotImplementedError

    def _mul_scalar(self, other):
        if other == 0:
            # Multiply by zero: return the zero matrix
            new = lil_matrix(self.shape, dtype=self.dtype)
        else:
            new = self.copy()
            # Multiply this scalar by every element.
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val*other for val in rowvals]
        return new

    def copy(self):
        from copy import deepcopy
        new = lil_matrix(self.shape, dtype=self.dtype)
        new.data = deepcopy(self.data)
        new.rows = deepcopy(self.rows)
        return new

    def reshape(self, shape):
        new = lil_matrix(shape, dtype=self.dtype)
        j_max = self.shape[1]
        for i, row in enumerate(self.rows):
            for j in row:
                new_r, new_c = np.unravel_index(i*j_max + j, shape)
                new[new_r, new_c] = self[i, j]
        return new

    def toarray(self, order=None, out=None):
        """See the docstring for `spmatrix.toarray`."""
        d = self._process_toarray_args(order, out)
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                d[i, j] = self.data[i][pos]
        return d


def isspmatrix_lil(x):
    return isinstance(x, lil_matrix)


def isspmatrix(x):
    return isinstance(x, spmatrix)
