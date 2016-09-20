from unittest import TestCase

from StateMapper import StateMapper


class TestStateMapper(TestCase):
    def test_getState(self):
        mapper = StateMapper([10], [0])

        actual = mapper.getState([3])[0]

        self.assertEquals(3,actual)

    def test_getState_for_multi_dimension(self):
        mapper = StateMapper([10, 10], [0, 0])

        actual = mapper.getState([3,5])

        self.assertListEqual([3,5], actual)


    def test_getState_for_multi_dimension_that_exceeds_range(self):
        mapper = StateMapper([10, 10], [0, 0])

        actual = mapper.getState([11,-1])

        self.assertListEqual([9,0], actual)

    def test_update_high_increased(self):
        mapper = StateMapper([10], [0])

        mapper.update([[11]])

        self.assertEquals(mapper.high[0], 11)


    def test_update_low_dicreased(self):
        mapper = StateMapper([10], [2])

        mapper.update([[1]])

        self.assertEquals(mapper.low[0], 1)

    def test_update_multi_dimensional_low(self):
        mapper = StateMapper([10, 10], [2, 2])

        mapper.update([[1,1]])

        self.assertEquals(mapper.low, [1,1])

    def test_update_multi_dimensional_high(self):
        mapper = StateMapper([10, 10], [2, 2])

        mapper.update([[11,11]])

        self.assertEquals(mapper.high, [11,11])

    def test_update_return_change_flag_when_any_range_changes(self):
        mapper = StateMapper([10, 10], [2, 3])

        actual = mapper.update([[11,1]])

        self.assertTrue(actual)

    def test_extend_high_by_delta(self):
        mapper = StateMapper([10],[0], '')
        mapper.extendRangePercent = 0.1
        mapper.extendVector()
        actual = mapper.high[0]

        self.assertEquals(actual,11)

    def test_extend_low_by_delta(self):
        mapper = StateMapper([10], [0], '')
        mapper.extendRangePercent = 0.1
        mapper.extendVector()
        actual = mapper.low[0]

        self.assertEquals(actual,-1)
