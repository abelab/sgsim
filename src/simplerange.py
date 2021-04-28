from typing import List
from space import CircularSpace

#
#     * a simple range class that does not support openness
#     * (open ends or closed ends).
#     * [x, x) is treated as a whole range [-∞, +∞].
#     *
#     * @param <K> type of the minimum and maximum value.
#     */


class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, item):
        return CircularSpace.is_ordered(self.start, True, item, self.end, False)

    def __str__(self):
        return "[{}, {})".format(self.start, self.end)

    def __repr__(self):
        return "[{}, {})".format(self.start, self.end)

    def contains(self, item):
        return item in self

    def containsExInc(self, key):
        return CircularSpace.is_ordered(self.start, False, key, self.end, True)

    # another ⊆ this
    def containsSimpleRange(self, another: 'Range'):
        if self.isWhole():
            return True
        if another.isWhole():
            return False
        return (self.contains(another.start)
                and self.containsExInc(another.end)
                and
                # exclude cases such as:
                #      this:  [=========)
                #   another: ====)   [====
                not another.contains(self.to))

    def hasIntersection(self, r: 'Range'):
        return (self.contains(r.start)
                or self.containsExInc(r.end)
                or r.contains(self.start)
                or r.containsExInc(self.end))

    def isWhole(self) -> bool:
        return self.start == self.end

    def retain(self, r: 'Range', removed: List['Range']) -> List['Range']:
        retains = []
        if r.isWhole():
            removed.append(self)
            return retains
        if not self.hasIntersection(r):
            retains.append(self)
            return retains
        # this: [             ]
        # r:    ......[........
        min = r.start if self.contains(r.start) else self.start
        max = r.end if self.contains(r.end) else self.end
        if CircularSpace.is_ordered_inclusive(self.start, min, max) and self.start != max:
            # this: [             ]
            # r:    ....[....]....
            if self.isWhole():  # simplify the results
                self.addIfNotPoint(retains, max, min)
            else:
                self.addIfNotPoint(retains, self.start, min)
                self.addIfNotPoint(retains, max, self.end)
            self.addIfNotPoint(removed, min, max)
        else:
            # this: [             ]
            # r:    ....]    [....
            self.addIfNotPoint(retains, max, min)
            if self.isWhole():  # simplify the results
                self.addIfNotPoint(removed, min, max)
            else:
                self.addIfNotPoint(removed, self.start, max)
                self.addIfNotPoint(removed, min, self.end)
        return retains

    def addIfNotPoint(self, lst: List['Range'], min, max) -> None:
        if min != max:
            lst.append(Range(min, max))

    @staticmethod
    def diffRanges(x: List['Range'], y: List['Range'], intersections: List['Range'] or None,
                   isPartialAllowed: bool) -> List['Range']:
        """
         * compute (x \ y) and (x ∩ y).
         * note that this method destroys x.
         *
         * @param x
         * @param y
         * @param intersections
         * @param isPartialAllowed true if elements in y that partially matches x
         *        are allowed.
         * @return x \ y
         """

        """
        /*
         * M' = {}  // 最終的にmatchするセグメントの集合
         * while M is not empty
         *   take m from M
         *   for (n in N)
         *     if (m と n が共通部分がある)
         *       m = m - n
         *       if (mが2つに分割)
         *         1つをMに追加し，残りをmとする．
         *       break if (m == empty);
         *   if (m is not empty)
         *    M’ = M' + m
         */
        """
        unsubtracted = list()
        while len(x) != 0:
            q: Range or None = x.pop(0)
            for r in y:
                if not isPartialAllowed:
                    if not q.contains(r):
                        continue
                    # retain = q - r
                    retain = q.retain(r, [])
                    if intersections:
                        intersections.append(r)
                else:
                    # intersect = q ∩ r
                    # retain = q - r
                    intersect = list()
                    #print("q={}".format(q))
                    #print("r={}".format(r))
                    retain = q.retain(r, intersect)
                    #print("retain={}".format(retain))
                    #print("intersect={}".format(intersect))
                    if intersections:
                        intersections.extend(intersect)
                if len(retain) == 2:
                    x.append(retain[1])
                    q = retain[0]
                elif len(retain) == 1:
                    q = retain[0]
                elif len(retain) == 0:
                    q = None
                    break
                else:
                    raise Exception("unexpected retain.size()")
            if q:
                unsubtracted.append(q)
        return unsubtracted
