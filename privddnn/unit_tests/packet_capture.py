import unittest
import os.path

from privddnn.utils.file_utils import read_jsonl_gz


class PacketCapture(unittest.TestCase):

    def test_max_prob_25(self):
        expected_results = read_jsonl_gz(os.path.join('..', 'distributed_results', 'uci_har', '14-02-2022', 'max_prob_50_25.jsonl.gz'))
        captured_results = read_jsonl_gz(os.path.join('..', 'distributed_results', 'uci_har', '14-02-2022', 'max_prob_50_25_processed_packets.jsonl.gz'))

        for expected, captured in zip(expected_results, captured_results):
            self.assertEquals(expected['exit_decision'], captured['exit_decision'])


if __name__ == '__main__':
    unittest.main()
