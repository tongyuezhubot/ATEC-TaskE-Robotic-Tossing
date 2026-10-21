import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.throwing.pose_cache import cache_key,sample_indices


class PoseCacheTests(unittest.TestCase):
    def test_dense_sampling_keeps_real_held_frames_without_duplicates(self):
        stages=["home"]*5+["check_hold"]*4+["plan"]*3+["swing"]*4
        self.assertEqual(sample_indices(stages,4),[0,4,8,12])
        actual=sample_indices(stages,4,True)
        self.assertEqual(actual,sorted(set(actual)))
        self.assertTrue(set(range(5,12)).issubset(actual))
        self.assertEqual([i for i in actual if stages[i]=="swing"],[12])

    def test_old_cache_identity_preserved_for_other_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path=(Path(directory)/"pose_data.hdf5").resolve();path.write_bytes(b"test")
            meta=Path(directory)/"run_config.json"
            meta.write_text(json.dumps({"throw_config":{"object_order":[2]}}))
            stat=path.stat()
            old=hashlib.sha256(f"spatial_patch_rgb_v1:{path}:{stat.st_size}:{stat.st_mtime_ns}:stride4".encode()).hexdigest()
            self.assertEqual(cache_key(path,4,2),old)
            meta.write_text(json.dumps({"throw_config":{"object_order":[1]}}))
            self.assertNotEqual(cache_key(path,4,2),old)
            self.assertEqual(cache_key(path,4),old)


if __name__=="__main__":
    unittest.main()
