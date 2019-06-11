import unittest
from mat2vec.processing import process


class ProcessTextTest(unittest.TestCase):

    def setUp(self):

        self.mtp = process.MaterialsTextProcessor()

        self.sentence_1 = "We measured 100 materials, including Ni(CO)4 and obtained very " \
                          "high Thermoelectric Figures of merit ZT. These results demonstrate " \
                          "the utility of Machine Learning methods for materials discovery."
        self.sentence_2 = "iron(II) was oxidized to obtain 5mg Ferrous Oxide"

    def test_tokenize(self):

        # test data
        sentences = [self.sentence_1] * 2 + [self.sentence_2] * 2
        split_oxidation = [True, False, True, False]
        keep_sentences = [True, False, True, False]

        # results
        tokens = [
            ["We measured 100 materials , including Ni(CO)4 and obtained very "
             "high Thermoelectric Figures of merit ZT .".split(),
             "These results demonstrate the utility of Machine Learning methods "
             "for materials discovery .".split()],
            "We measured 100 materials , including Ni(CO)4 and obtained very "
            "high Thermoelectric Figures of merit ZT . These results demonstrate "
            "the utility of Machine Learning methods for materials discovery .".split(),
            ["iron (II) was oxidized to obtain 5 mg Ferrous Oxide".split()],
            "iron(II) was oxidized to obtain 5 mg Ferrous Oxide".split(),
        ]

        # running the tests
        for sent, toks, so, ks in zip(sentences, tokens, split_oxidation, keep_sentences):
            self.assertListEqual(self.mtp.tokenize(sent, split_oxidation=so, keep_sentences=ks), toks)

    def test_process(self):

        # test data
        sentences = [self.sentence_1] * 3 + [self.sentence_2] * 2
        matnorm = [True, False, True, True, False]
        convert_nums = [False, True, True, False, True]
        exclude_punct = [True, False, False, True, False]
        make_phrases = [False, False, True, False, True]

        # result
        processed = [
            "we measured 100 materials including C4NiO4 and obtained very "
            "high thermoelectric figures of merit ZT these results demonstrate "
            "the utility of machine learning methods for materials discovery".split(),
            "we measured <nUm> materials , including Ni(CO)4 and obtained very "
            "high thermoelectric figures of merit ZT . these results demonstrate "
            "the utility of machine learning methods for materials discovery .".split(),
            "we measured <nUm> materials , including C4NiO4 and obtained very "
            "high thermoelectric_figures_of_merit ZT . these results_demonstrate_the_utility "
            "of machine_learning methods for materials discovery .".split(),
            "iron (II) was oxidized to obtain 5 mg ferrous oxide".split(),
            "iron (II) was oxidized to obtain <nUm> mg ferrous oxide".split()
        ]
        materials = [
            [("Ni(CO)4", "C4NiO4")],
            [("Ni(CO)4", "C4NiO4")],
            [("Ni(CO)4", "C4NiO4")],
            [("iron", "Fe")],
            [("iron", "Fe")]
        ]

        # running the tests
        for s, mn, cn, ep, mp, pr, mm in zip(
                sentences, matnorm, convert_nums, exclude_punct, make_phrases, processed, materials):

            proc, mats = self.mtp.process(
                s, normalize_materials=mn, convert_num=cn, exclude_punct=ep, make_phrases=mp)
            self.assertListEqual(proc, pr)
            self.assertListEqual(mats, mm)

    def test_is_number(self):

        self.assertTrue(self.mtp.is_number("-5"))
        self.assertTrue(self.mtp.is_number("-5.5"))
        self.assertTrue(self.mtp.is_number("123.4"))
        self.assertTrue(self.mtp.is_number("1,000,000"))

        self.assertFalse(self.mtp.is_number("not a number"))
        self.assertFalse(self.mtp.is_number("with23numbers"))
        self.assertFalse(self.mtp.is_number("23.54b"))
        self.assertFalse(self.mtp.is_number("-23a"))

    def test_is_simple_formula(self):

        self.assertTrue(self.mtp.is_simple_formula("CoO"))
        self.assertTrue(self.mtp.is_simple_formula("H2O"))
        self.assertTrue(self.mtp.is_simple_formula("C(OH)2"))
        self.assertTrue(self.mtp.is_simple_formula("C(OH)2Si"))
        self.assertTrue(self.mtp.is_simple_formula("Ni0.5(CO)2"))

        self.assertFalse(self.mtp.is_simple_formula("Ad2Al"))
        self.assertFalse(self.mtp.is_simple_formula("123"))
        self.assertFalse(self.mtp.is_simple_formula("some other text"))
        self.assertFalse(self.mtp.is_simple_formula("Ni0.5(CO)2)"))

    def test_get_ordered_integer_formula(self):

        self.assertEqual(self.mtp.get_ordered_integer_formula({"Al": 1, "O": 1.5}), "Al2O3")
        self.assertEqual(self.mtp.get_ordered_integer_formula({"C": 0.04, "Al": 0.02, "O": 0.5}), "AlC2O25")

    def test_normalized_formula(self):

        self.assertEqual(self.mtp.normalized_formula("SiO2"), "O2Si")
        self.assertEqual(self.mtp.normalized_formula("Ni(CO)4"), "C4NiO4")
        self.assertEqual(self.mtp.normalized_formula("Ni(CO)4SiO2"), "C4NiO6Si")

    def test_remove_accent(self):

        self.assertEqual(self.mtp.remove_accent("Louis Eugène Félix Néel"), "Louis Eugene Felix Neel")
        self.assertEqual(self.mtp.remove_accent("ångström"), "angstrom")
        self.assertEqual(self.mtp.remove_accent("Å"), "Å")

    def test_is_element(self):

        for elem in self.mtp.ELEMENTS[:100]:
            self.assertTrue(self.mtp.is_element(elem))

        self.assertFalse(self.mtp.is_element("not an element"))
        self.assertFalse(self.mtp.is_element("A"))
        self.assertFalse(self.mtp.is_element("Dc"))
        self.assertFalse(self.mtp.is_element("14"))
        self.assertFalse(self.mtp.is_element("We"))
