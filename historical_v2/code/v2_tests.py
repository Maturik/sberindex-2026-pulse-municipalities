"""Meaningful artificial invariants; no economic outcomes or proprietary core."""
import unittest
import numpy as np
from v2_forecasting import calendar_profiles, theta_lite
from v2_detectors import DETECTORS, trace, bocpd_statistics
from v2_metrics import metric_table, rank_methods


class ProtocolInvariants(unittest.TestCase):
    def test_calendar_does_not_read_2024(self):
        first = np.tile(np.arange(24)+100., (6, 1))
        second = first.copy()
        second[:, 12:] = 1e12
        np.testing.assert_array_equal(calendar_profiles(first, ["a"]*6, ["a"]),
                                      calendar_profiles(second, ["a"]*6, ["a"]))

    def test_theta_closed_formula(self):
        y = np.arange(1.,13.)
        level = 1.
        for value in y[1:]:
            level = .2*value+.8*level
        expected = level+.5*(np.arange(1,13)-1+(1-.8**12)/.2)
        np.testing.assert_allclose(theta_lite(y), expected)

    def test_detectors_are_prefix_causal(self):
        z = np.array([0., .1, -.2, 0., 2., 3., 1., -3., 5., 0., .2, 0.])
        for name in DETECTORS:
            full = trace(z, name, .4)
            for length in range(1,len(z)+1):
                short = trace(z[:length],name,.4)
                np.testing.assert_allclose(full[0][:length],short[0])
                np.testing.assert_array_equal(full[1][:length],short[1])

    def test_cusum_resets_both_sides(self):
        statistic, alarm = trace([4.,0.,-4.,0.],"ResetCUSUM",2.)
        np.testing.assert_array_equal(alarm,[True,False,True,False])
        np.testing.assert_allclose(statistic,[3.5,0.,3.5,0.])

    def test_bocpd_score_is_not_constant_hazard(self):
        score = bocpd_statistics([0.,0.,0.,4.,4.,4.])
        self.assertTrue(np.all((score>=0)&(score<=1)))
        self.assertGreater(np.ptp(score),.01)

    def test_metric_is_macro_not_pooled_scale(self):
        rows=[]
        for method in ("a","b"):
            for h in (1,3,6):
                for tid,scale in ((1,1.),(2,1000.)):
                    rows.append(dict(split="validation",method=method,horizon=h,territory_id=tid,
                        category="a",prefix_n=12,actual=scale,prediction=scale*(1.1 if method=="a" else 1.2),
                        reference_scale=scale,mase_scale=scale))
        table=metric_table(rows)
        winner,scores=rank_methods(table,["a","b"])
        self.assertEqual(winner,"a")
        self.assertAlmostEqual(scores["a"],.1)
        self.assertAlmostEqual(scores["b"],.2)


if __name__ == "__main__":
    unittest.main()
