#!/usr/bin/env python3
import unittest

from deterministic_validate import validate_pair


class DeterministicValidatorTests(unittest.TestCase):
    def assert_status(self, expected, name, original, perturbed):
        self.assertEqual(validate_pair(name, original, perturbed).status, expected)

    def test_relevant_emojis(self):
        self.assert_status("pass", "A1_emoji_relevant", "Climate policy changed.", "Climate policy changed. 🌍🔥")
        self.assert_status("fail", "A1_emoji_relevant", "Climate policy changed.", "Climate policy changed. 🌍")

    def test_disruptive_emoji(self):
        self.assert_status("pass", "A1_emoji_disruptive", "The policy changed", "The 🚨 policy changed")

    def test_hashtags(self):
        original = "The policy changed"
        perturbed = original + " #Policy #Change #Government #Breaking #Viral #MustRead"
        self.assert_status("pass", "A2_hashtagification", original, perturbed)

    def test_mapped_changes(self):
        self.assert_status("pass", "C2_leetspeak", "safe", "s4f3")
        self.assert_status("fail", "C2_leetspeak", "safe", "surf")

    def test_word_splitting(self):
        self.assert_status("pass", "C3_word_splitting", "government vaccination", "govern ment vaccin ation")
        self.assert_status("fail", "C3_word_splitting", "government vaccination", "govern ment vaccination")

    def test_presupposition(self):
        self.assert_status("pass", "E2_presupposition", "The policy changed.", "As everyone knows, The policy changed.")

    def test_not_applicable(self):
        self.assert_status("not_applicable", "P_llm_rewrite_high", "A claim", "A rewritten claim")


if __name__ == "__main__":
    unittest.main()
