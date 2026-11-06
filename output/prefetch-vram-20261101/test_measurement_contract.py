"""Stdlib regression for WSL process-identity changes; never uses GPU."""
import importlib.util
import unittest
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('measurement_analyze',H/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def sample(at,cpu,identity):
    return {'at_ns':at,'processes':[{'identity':identity,'cpu_s':cpu,'rss_bytes':100}]}
class IdentityContract(unittest.TestCase):
    def test_stable_kernel_ticks_preserve_cpu_delta(self):
        raw={'load_start_ns':100,'load_end_ns':200,'resource_samples':[sample(90,1,'42:900'),sample(150,2,'42:900'),sample(210,3,'42:900')]}
        r=a.resources(raw)
        self.assertTrue(r['cpu_process_identity_stable'])
        self.assertEqual(r['cpu_s_sampled'],2)
    def test_shifted_creation_wallclock_is_missing_not_zero_cpu(self):
        raw={'load_start_ns':100,'load_end_ns':200,'resource_samples':[sample(90,1,'42:1793537000.1'),sample(150,2,'42:1793537004.1'),sample(210,3,'42:1793537004.1')]}
        r=a.resources(raw)
        self.assertFalse(r['cpu_process_identity_stable'])
        self.assertIsNone(r['cpu_s_sampled'])
        self.assertNotEqual(r['cpu_s_sampled'],0)
if __name__=='__main__': unittest.main()
