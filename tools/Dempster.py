"""
A framework for performing computations in the Dempster-Shafer theory.
"""

from __future__ import print_function
from itertools import chain, combinations
from functools import partial, reduce
from operator import mul
from math import log, fsum, sqrt
from random import random, shuffle, uniform
import sys
import numpy as np

try:
    import numpy

    try:
        from scipy.stats import chi2
        from scipy.optimize import fmin_cobyla
    except:
        print('SciPy not found: some features will not work.', file=sys.stderr)
except:
    print('NumPy not found: some features will not work.', file=sys.stderr)


class MassFunction(dict):
    """
    A Dempster-Shafer mass function (basic probability assignment) based on a dictionary.

    Both normalized and unnormalized mass functions are supported.
    The underlying frame of discernment is assumed to be discrete.

    Hypotheses and their associated mass values can be added/changed/removed using the standard dictionary methods.
    Each hypothesis can be an arbitrary sequence which is automatically converted to a 'frozenset', meaning its elements must be hashable.
    """

    def __init__(self, source=None):
        """
        Creates a new mass function.

        If 'source' is not None, it is used to initialize the mass function.
        It can either be a dictionary mapping hypotheses to non-negative mass values
        or an iterable containing tuples consisting of a hypothesis and a corresponding mass value.
        """
        if source != None:
            if isinstance(source, dict):
                source = source.items()
            for (h, v) in source:
                self[h] += v

    @staticmethod  # 可以不创建实例就可以调用方法
    def _convert(hypothesis):
        """Convert hypothesis to a 'frozenset' in order to make it hashable."""
        if isinstance(hypothesis, frozenset):
            return hypothesis
        else:
            return frozenset(hypothesis)  # 冻结后集合不能再添加或删除任何元素

    @staticmethod
    def gbt(likelihoods, normalization=True, sample_count=None):
        """
        Constructs a mass function using the generalized Bayesian theorem.
        For more information, see Smets. 1993. Belief functions:
        The disjunctive rule of combination and the generalized Bayesian theorem. International Journal of Approximate Reasoning.

        'likelihoods' specifies the conditional plausibilities for a set of singleton hypotheses.
        It can either be a dictionary mapping singleton hypotheses to plausibilities or an iterable
        containing tuples consisting of a singleton hypothesis and a corresponding plausibility value.

        'normalization' determines whether the resulting mass function is normalized, i.e., whether m({}) == 0.

        If 'sample_count' is not None, the true mass function is approximated using the specified number of samples.
        """
        m = MassFunction()
        if isinstance(likelihoods, dict):
            likelihoods = list(likelihoods.items())
        # filter trivial likelihoods 0 and 1
        ones = [h for (h, l) in likelihoods if l >= 1.0]
        likelihoods = [(h, l) for (h, l) in likelihoods if 0.0 < l < 1.0]
        if sample_count == None:  # deterministic
            def traverse(m, likelihoods, ones, index, hyp, mass):
                if index == len(likelihoods):
                    m[hyp + ones] = mass
                else:
                    traverse(m, likelihoods, ones, index + 1, hyp + [likelihoods[index][0]],
                             mass * likelihoods[index][1])
                    traverse(m, likelihoods, ones, index + 1, hyp, mass * (1.0 - likelihoods[index][1]))

            traverse(m, likelihoods, ones, 0, [], 1.0)
            if normalization:
                m.normalize()
        else:  # Monte-Carlo
            if normalization:
                empty_mass = reduce(mul, [1.0 - l[1] for l in likelihoods], 1.0)
            for _ in range(sample_count):
                rv = [random() for _ in range(len(likelihoods))]
                subtree_mass = 1.0
                hyp = set(ones)
                for k in range(len(likelihoods)):
                    l = likelihoods[k][1]
                    p_t = l * subtree_mass
                    p_f = (1.0 - l) * subtree_mass
                    if normalization and not hyp:  # avoid empty hypotheses in the normalized case
                        p_f -= empty_mass
                    if p_t > rv[k] * (p_t + p_f):
                        hyp.add(likelihoods[k][0])
                    else:
                        subtree_mass *= 1 - l  # only relevant for the normalized empty case
                m[hyp] += 1.0 / sample_count
        return m

    @staticmethod
    def from_bel(bel):  # 从相应的置信度函数创建质量函数。
        """
        Creates a mass function from a corresponding belief function.

        'bel' is a dictionary mapping hypotheses to belief values (like the dictionary returned by 'bel(None)').
        """
        m = MassFunction()
        for h1 in bel.keys():
            v = fsum([bel[h2] * (-1) ** (len(h1 - h2)) for h2 in powerset(h1)])
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
        return m

    @staticmethod
    def from_pl(pl):
        """
        Creates a mass function from a corresponding plausibility function.

        'pl' is a dictionary mapping hypotheses to plausibility values (like the dictionary returned by 'pl(None)').
        """
        frame = max(pl.keys(), key=len)
        bel_theta = pl[frame]
        bel = {frozenset(frame - h): bel_theta - v for (h, v) in
               pl.items()}  # follows from bel(-A) = bel(frame) - pl(A)
        return MassFunction.from_bel(bel)

    @staticmethod
    def from_q(q):
        """
        Creates a mass function from a corresponding commonality function.

        'q' is a dictionary mapping hypotheses to commonality values (like the dictionary returned by 'q(None)').
        """
        m = MassFunction()
        frame = max(q.keys(), key=len)
        for h1 in q.keys():
            v = fsum([q[h1 | h2] * (-1) ** (len(h2 - h1)) for h2 in powerset(frame - h1)])
            if v > 0:
                m[h1] = v
        mass_sum = fsum(m.values())
        if mass_sum < 1.0:
            m[frozenset()] = 1.0 - mass_sum
        return m

    def __missing__(self, key):
        """Return 0 mass for hypotheses that are not contained."""
        return 0.0

    def __copy__(self):
        c = MassFunction()
        for k, v in self.items():
            c[k] = v
        return c

    def copy(self):
        """Creates a shallow copy of the mass function."""
        return self.__copy__()

    def __contains__(self, hypothesis):
        return dict.__contains__(self, MassFunction._convert(hypothesis))

    def __getitem__(self, hypothesis):
        return dict.__getitem__(self, MassFunction._convert(hypothesis))

    def __setitem__(self, hypothesis, value):
        """
        Adds or updates the mass value of a hypothesis.

        'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        In case of a negative mass value, a ValueError is raised.
        """
        if value < 0.0:
            raise ValueError("mass value is negative: %f" % value)
        dict.__setitem__(self, MassFunction._convert(hypothesis), value)

    def __delitem__(self, hypothesis):
        return dict.__delitem__(self, MassFunction._convert(hypothesis))

    def frame(self):
        """
        Returns the frame of discernment of the mass function as a 'frozenset'.

        The frame of discernment is the union of all contained hypotheses.
        In case the mass function does not contain any hypotheses, an empty set is returned.
        """
        if not self:
            return frozenset()
        else:
            return frozenset.union(*self.keys())


    # 返回焦点集
    def focal(self):
        """
        Returns the set of all focal hypotheses.

        A focal hypothesis has a mass value greater than 0.
        """
        return {h for (h, v) in self.items() if v > 0}

    # 返回焦点集里出现的点
    def core(self, *mass_functions):
        """
        Returns the core of one or more mass functions as a 'frozenset'.

        The core of a single mass function is the union of all its focal hypotheses.
        In case a mass function does not contain any focal hypotheses, its core is an empty set.
        If multiple mass functions are given, their combined core (intersection of all single cores) is returned.
        """
        if mass_functions:
            return frozenset.intersection(self.core(), *[m.core() for m in mass_functions])
        else:
            focal = self.focal()
            if not focal:
                return frozenset()
            else:
                return frozenset.union(*focal)

    def all(self):
        """Returns an iterator over all subsets of the frame of discernment, including the empty set."""
        return powerset(self.frame())

    # 计算bel值
    def bel(self, hypothesis=None):
        """
        Computes either the belief of 'hypothesis' or the entire belief function (hypothesis=None).

        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective belief values is returned.
        Otherwise, the belief of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:  # 若hypothesis为空，则返回的是焦点集中所包含的元素所构成的所有幂集的bel
            return {h: self.bel(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:  # 集合里面一个东西都没有的话，返回0
                return 0.0
            else:  # fsum是精准的浮点数求和，issuperset()方法用于判断指定集合的所有元素是否都包含在原始的集合中
                return fsum([v for (h, v) in self.items() if h and hypothesis.issuperset(h)])

    def get_Ed(self):
        Ed = - fsum([v * log(v / (2 ** len(h) - 1),2) for (h,v) in self.items() if v > 0])
        return Ed

    def get_Ep(self):
        Ep = -fsum([])

    # 计算pl值
    def pl(self, hypothesis=None):
        """
        Computes either the plausibility of 'hypothesis' or the entire plausibility function (hypothesis=None).

        If 'hypothesis' is None (default), a dictionary mapping hypotheses to their respective plausibility values is returned.
        Otherwise, the plausibility of 'hypothesis' is returned.
        In this case, 'hypothesis' is automatically converted to a 'frozenset' meaning its elements must be hashable.
        """
        if hypothesis is None:
            return {h: self.pl(h) for h in powerset(self.core())}
        else:
            hypothesis = MassFunction._convert(hypothesis)
            if not hypothesis:
                return 0.0
            else:
                return fsum([v for (h, v) in self.items() if hypothesis & h])


    def __and__(self, mass_function):
        """Shorthand for 'combine_conjunctive(mass_function)'."""
        return self.combine_conjunctive(mass_function)

    def __or__(self, mass_function):
        """Shorthand for 'combine_disjunctive(mass_function)'."""
        return self.combine_disjunctive(mass_function)

    def __str__(self):
        hyp = sorted([(v, h) for (h, v) in self.items()], reverse=True)
        return "{" + "; ".join([str(set(h)) + ":" + str(v) for (v, h) in hyp]) + "}"

    def __mul__(self, scalar):
        if not isinstance(scalar, float):
            raise TypeError('Can only multiply by a float value.')
        m = MassFunction()
        for (h, v) in self.items():
            m[h] = v * scalar
        return m

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, m):
        if not isinstance(m, MassFunction):
            raise TypeError('Can only add two mass functions.')
        result = self.copy()
        for (h, v) in m.items():
            result[h] += v
        return result


    def combine_conjunctive(self, mass_function, normalization=True, sample_count=None, importance_sampling=False):
        """
        Conjunctively combines the mass function with another mass function and returns the combination as a new mass function.

        The other mass function is assumed to be defined over the same frame of discernment.
        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions that are iteratively combined.

        If the mass functions are flatly contracting or if one of the mass functions is empty, an empty mass function is returned.

        'normalization' determines whether the resulting mass function is normalized (default is True).

        If 'sample_count' is not None, the true combination is approximated using the specified number of samples.
        In this case, 'importance_sampling' determines the method of approximation (only if normalization=True, otherwise 'importance_sampling' is ignored).
        The default method (importance_sampling=False) independently generates samples from both mass functions and computes their intersections.
        If importance_sampling=True, importance sampling is used to avoid empty intersections, which leads to a lower approximation error but is also slower.
        This method should be used if there is significant evidential conflict between the mass functions.
        """
        return self._combine(mass_function, rule=lambda s1, s2: s1 & s2, normalization=normalization,
                             sample_count=sample_count, importance_sampling=importance_sampling)



    def _combine(self, mass_function, rule, normalization, sample_count, importance_sampling):
        """Helper method for combining two or more mass functions."""
        combined = self
        if isinstance(mass_function, MassFunction):
            mass_function = [mass_function]  # wrap single mass function
        for m in mass_function:
            if not isinstance(m, MassFunction):
                raise TypeError(
                    "expected type MassFunction but got %s; make sure to use keyword arguments for anything other than mass functions" % type(
                        m))
            if sample_count == None:
                combined = combined._combine_deterministic(m, rule)
            else:
                if importance_sampling and normalization:
                    combined = combined._combine_importance_sampling(m, sample_count)
                else:
                    combined = combined._combine_direct_sampling(m, rule, sample_count)
        if normalization:
            return combined.normalize()
        else:
            return combined

    def _combine_deterministic(self, mass_function, rule):
        """Helper method for deterministically combining two mass functions."""
        combined = MassFunction()
        for (h1, v1) in self.items():
            for (h2, v2) in mass_function.items():
                combined[rule(h1, h2)] += v1 * v2
        return combined




    def conflict(self, mass_function, sample_count=None):
        """
        Calculates the weight of conflict between two or more mass functions.

        If 'mass_function' is not of type MassFunction, it is assumed to be an iterable containing multiple mass functions.

        The weight of conflict is computed as the (natural) logarithm of the normalization constant in Dempster's rule of combination.
        Returns infinity in case the mass functions are flatly contradicting.
        """
        # compute full conjunctive combination (could be more efficient)
        m = self.combine_conjunctive(mass_function, normalization=False, sample_count=sample_count)
        print("随便输出", m)
        empty = m[frozenset()]
        m_sum = fsum(m.values())
        diff = m_sum - empty
        if diff == 0.0:
            return float('inf')
        else:
            return -log(diff)

    def normalize(self):
        """
        Normalizes the mass function in-place.

        Sets the mass value of the empty set to 0 and scales all other values such that their sum equals 1.
        For convenience, the method returns 'self'.
        """
        if frozenset() in self:
            del self[frozenset()]
        mass_sum = fsum(self.values())
        if mass_sum != 1.0:
            for (h, v) in self.items():
                self[h] = v / mass_sum
        return self





def powerset(iterable):
    """
    Returns an iterator over the power set of 'set'.

    'set' is an arbitrary iterator over hashable elements.
    All returned subsets are of type 'frozenset'.
    """
    # combinations从iterable中选r个单位进行组合后用chain将其分成{'',''}格式后变成冻结集合
    return map(frozenset, chain.from_iterable(combinations(iterable, r) for r in range(len(iterable) + 1)))

