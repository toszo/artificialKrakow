from unittest import TestCase

from Discretization import Discretization


class TestDiscretization(TestCase):
    def test_save(self):
        self.fail()

    def test_load(self):
        self.fail()

    def test_getState(self):
        discretizer = Discretization([10],[0])

        actual = discretizer.getState([3])[0]

        self.assertEquals(3,actual)

    def test_getState_for_multi_dimension(self):
        discretizer = Discretization([10,10],[0,0])

        actual = discretizer.getState([3,5])

        self.assertListEqual([3,5], actual)


    def test_getState_for_multi_dimension_that_exceeds_range(self):
        discretizer = Discretization([10,10],[0,0])

        actual = discretizer.getState([11,-1])

        self.assertListEqual([9,0], actual)

    def test_update_high_increased(self):
        discretizer = Discretization([10],[0])

        discretizer.update([[11]])

        self.assertEquals(discretizer.high[0], 11)


    def test_update_low_dicreased(self):
        discretizer = Discretization([10], [2])

        discretizer.update([[1]])

        self.assertEquals(discretizer.low[0], 1)

    def test_update_multi_dimensional_low(self):
        discretizer = Discretization([10,10], [2,2])

        discretizer.update([[1,1]])

        self.assertEquals(discretizer.low, [1,1])

    def test_update_multi_dimensional_high(self):
        discretizer = Discretization([10,10], [2,2])

        discretizer.update([[11,11]])

        self.assertEquals(discretizer.high, [11,11])

    def test_update_return_change_flag_when_any_range_changes(self):
        discretizer = Discretization([10,10], [2,3])

        actual = discretizer.update([[11,1]])

        self.assertTrue(actual)
