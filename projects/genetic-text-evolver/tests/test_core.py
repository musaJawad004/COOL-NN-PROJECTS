import random
import unittest

from model.vocabulary import Vocabulary
from train import make_typo


class VocabularyTests(unittest.TestCase):
    def test_round_trip(self):
        vocabulary = Vocabulary.build(["hello, world!"])
        self.assertEqual(vocabulary.decode(vocabulary.encode("hello, world!")), "hello, world!")

    def test_unknown_character_is_ignored_when_decoded(self):
        vocabulary = Vocabulary.build(["abc"])
        encoded = vocabulary.encode("a🙂c")
        self.assertEqual(vocabulary.decode(encoded), "ac")


class AugmentationTests(unittest.TestCase):
    def test_typo_changes_text(self):
        source = "testing"
        self.assertNotEqual(make_typo(source, random.Random(2)), source)


if __name__ == "__main__":
    unittest.main()
