import regex
import string
import unidecode
from os import path
from monty.fractions import gcd_float

from chemdataextractor.doc import Paragraph
from gensim.models.phrases import Phraser

from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition, CompositionError

PHRASER_PATH = path.join(path.dirname(__file__), "models/phraser.pkl")

__author__ = "Vahe Tshitoyan"
__credits__ = "John Dagdelen, Leigh Weston, Anubhav Jain"
__copyright__ = "Copyright 2018 - 2019, Materials Intelligence."
__version__ = "0.0.3"
__maintainer__ = "John Dagdelen"
__email__ = "vahe.tshitoyan@gmail.com, jdagdelen@berkeley.edu"
__date__ = "June 10, 2019"


class MaterialsTextProcessor:
    """
    Materials Science Text Processing Tools.
    """
    ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
                "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
                "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
                "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
                "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

    ELEMENT_NAMES = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine",
                     "neon", "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
                     "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
                     "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine",
                     "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium",
                     "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium",
                     "iodine", "xenon", "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
                     "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium",
                     "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium",
                     "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium", "astatine",
                     "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium",
                     "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
                     "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium",
                     "hassium", "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
                     "moscovium", "livermorium", "tennessine", "oganesson", "ununennium"]

    ELEMENTS_AND_NAMES = ELEMENTS + ELEMENT_NAMES + [en.capitalize() for en in ELEMENT_NAMES]
    ELEMENTS_NAMES_UL = ELEMENT_NAMES + [en.capitalize() for en in ELEMENT_NAMES]

    # Elemement with the valence state in parenthesis.
    ELEMENT_VALENCE_IN_PAR = regex.compile(r"^("+r"|".join(ELEMENTS_AND_NAMES) +
                                           r")(\(([IV|iv]|[Vv]?[Ii]{0,3})\))$")
    ELEMENT_DIRECTION_IN_PAR = regex.compile(r"^(" + r"|".join(ELEMENTS_AND_NAMES) + r")(\(\d\d\d\d?\))")

    # Exactly IV, VI or has 2 consecutive II, or roman in parenthesis: is not a simple formula.
    VALENCE_INFO = regex.compile(r"(II+|^IV$|^VI$|\(IV\)|\(V?I{0,3}\))")

    SPLIT_UNITS = ["K", "h", "V", "wt", "wt.", "MHz", "kHz", "GHz", "Hz", "days", "weeks",
                   "hours", "minutes", "seconds", "T", "MPa", "GPa", "at.", "mol.",
                   "at", "m", "N", "s-1", "vol.", "vol", "eV", "A", "atm", "bar",
                   "kOe", "Oe", "h.", "mWcm−2", "keV", "MeV", "meV", "day", "week", "hour",
                   "minute", "month", "months", "year", "cycles", "years", "fs", "ns",
                   "ps", "rpm", "g", "mg", "mAcm−2", "mA", "mK", "mT", "s-1", "dB",
                   "Ag-1", "mAg-1", "mAg−1", "mAg", "mAh", "mAhg−1", "m-2", "mJ", "kJ",
                   "m2g−1", "THz", "KHz", "kJmol−1", "Torr", "gL-1", "Vcm−1", "mVs−1",
                   "J", "GJ", "mTorr", "bar", "cm2", "mbar", "kbar", "mmol", "mol", "molL−1",
                   "MΩ", "Ω", "kΩ", "mΩ", "mgL−1", "moldm−3", "m2", "m3", "cm-1", "cm",
                   "Scm−1", "Acm−1", "eV−1cm−2", "cm-2", "sccm", "cm−2eV−1", "cm−3eV−1",
                   "kA", "s−1", "emu", "L", "cmHz1", "gmol−1", "kVcm−1", "MPam1",
                   "cm2V−1s−1", "Acm−2", "cm−2s−1", "MV", "ionscm−2", "Jcm−2", "ncm−2",
                   "Jcm−2", "Wcm−2", "GWcm−2", "Acm−2K−2", "gcm−3", "cm3g−1", "mgl−1",
                   "mgml−1", "mgcm−2", "mΩcm", "cm−2s−1", "cm−2", "ions", "moll−1",
                   "nmol", "psi", "mol·L−1", "Jkg−1K−1", "km", "Wm−2", "mass", "mmHg",
                   "mmmin−1", "GeV", "m−2", "m−2s−1", "Kmin−1", "gL−1", "ng", "hr", "w",
                   "mN", "kN", "Mrad", "rad", "arcsec", "Ag−1", "dpa", "cdm−2",
                   "cd", "mcd", "mHz", "m−3", "ppm", "phr", "mL", "ML", "mlmin−1", "MWm−2",
                   "Wm−1K−1", "Wm−1K−1", "kWh", "Wkg−1", "Jm−3", "m-3", "gl−1", "A−1",
                   "Ks−1", "mgdm−3", "mms−1", "ks", "appm", "ºC", "HV", "kDa", "Da", "kG",
                   "kGy", "MGy", "Gy", "mGy", "Gbps", "μB", "μL", "μF", "nF", "pF", "mF",
                   "A", "Å", "A˚", "μgL−1"]

    NR_BASIC = regex.compile(r"^[+-]?\d*\.?\d+\(?\d*\)?+$", regex.DOTALL)
    NR_AND_UNIT = regex.compile(r"^([+-]?\d*\.?\d+\(?\d*\)?+)([\p{script=Latin}|Ω|μ]+.*)", regex.DOTALL)

    PUNCT = list(string.punctuation) + ["\"", "“", "”", "≥", "≤", "×"]

    def __init__(self, phraser_path=PHRASER_PATH):
        self.elem_name_dict = {en: es for en, es in zip(self.ELEMENT_NAMES, self.ELEMENTS)}
        self.phraser = Phraser.load(phraser_path)

    def tokenize(self, text, split_oxidation=True, keep_sentences=True):
        """Converts a string to a list tokens (words) using a modified chemdataextractor tokenizer.

        Adds a few fixes for inorganic materials science, such as splitting common units from numbers
        and splitting the valence state.

        Args:
            text: input text as a string
            split_oxidation: if True, will split the oxidation state from the element, e.g. iron(II)
                will become iron (II), same with Fe(II), etc.
            keep_sentences: if False, will disregard the sentence structure and return tokens as a
                single list of strings. Otherwise returns a list of lists, each sentence separately.

        Returns:
            A list of strings if keep_sentence is False, otherwise a list of list of strings, which each
            list corresponding to a single sentence.
        """
        def split_token(token, so=split_oxidation):
            """Processes a single token, in case it needs to be split up.

            There are 2 cases when the token is split: A number with a common unit, or an
            element with a valence state.

            Args:
                token: The string to be processed.
                so: If True, split the oxidation (valence) string. Units are always split.

            Returns:
                A list of strings.
            """
            elem_with_valence = self.ELEMENT_VALENCE_IN_PAR.match(token) if so else None
            nr_unit = self.NR_AND_UNIT.match(token)
            if nr_unit is not None and nr_unit.group(2) in self.SPLIT_UNITS:
                # Splitting the unit from number, e.g. "5V" -> ["5", "V"].
                return [nr_unit.group(1), nr_unit.group(2)]
            elif elem_with_valence is not None:
                # Splitting element from it"s valence state, e.g. "Fe(II)" -> ["Fe", "(II)"].
                return [elem_with_valence.group(1), elem_with_valence.group(2)]
            else:
                return [token]

        cde_p = Paragraph(text)
        tokens = cde_p.tokens
        toks = []
        for sentence in tokens:
            if keep_sentences:
                toks.append([])
                for tok in sentence:
                    toks[-1] += split_token(tok.text, so=split_oxidation)
            else:
                for tok in sentence:
                    toks += split_token(tok.text, so=split_oxidation)
        return toks

    def process(self, tokens, exclude_punct=False, convert_num=True, normalize_materials=True, remove_accents=True,
                make_phrases=False, split_oxidation=True):
        """Processes a pre-tokenized list of strings or a string.

        Selective lower casing, material normalization, etc.

        Args:
            tokens: A list of strings or a string. If a string is supplied, will use the
                tokenize method first to split it into a list of token strings.
            exclude_punct: Bool flag to exclude all punctuation.
            convert_num: Bool flag to convert numbers (selectively) to <nUm>.
            normalize_materials: Bool flag to normalize all simple material formula.
            remove_accents: Bool flag to remove accents, e.g. Néel -> Neel.
            make_phrases: Bool flag to convert single tokens to common materials science phrases.
            split_oxidation: Only used if string is supplied, see docstring for tokenize method.

        Returns:
            A (processed_tokens, material_list) tuple. processed_tokens is a list of strings,
            whereas material_list is a list of (original_material_string, normalized_material_string)
            tuples.
        """

        if not isinstance(tokens, list):  # If it"s a string.
            return self.process(self.tokenize(
                tokens, split_oxidation=split_oxidation, keep_sentences=False),
                exclude_punct=exclude_punct,
                convert_num=convert_num,
                normalize_materials=normalize_materials,
                remove_accents=remove_accents,
                make_phrases=make_phrases
            )

        processed, mat_list = [], []

        for i, tok in enumerate(tokens):
            if exclude_punct and tok in self.PUNCT:  # Punctuation.
                continue
            elif convert_num and self.is_number(tok):  # Number.
                # Replace all numbers with <nUm>, except if it is a crystal direction (e.g. "(111)").
                try:
                    if tokens[i - 1] == "(" and tokens[i + 1] == ")" \
                            or tokens[i - 1] == "〈" and tokens[i + 1] == "〉":
                        pass
                    else:
                        tok = "<nUm>"
                except IndexError:
                    tok = "<nUm>"
            elif tok in self.ELEMENTS_NAMES_UL:  # Chemical element name.
                # Add as a material mention.
                mat_list.append((tok, self.elem_name_dict[tok.lower()]))
                tok = tok.lower()
            elif self.is_simple_formula(tok):  # Simple chemical formula.
                normalized_formula = self.normalized_formula(tok)
                mat_list.append((tok, normalized_formula))
                if normalize_materials:
                    tok = normalized_formula
            elif (len(tok) == 1 or (len(tok) > 1 and tok[0].isupper() and tok[1:].islower())) \
                    and tok not in self.ELEMENTS and tok not in self.SPLIT_UNITS \
                    and self.ELEMENT_DIRECTION_IN_PAR.match(tok) is None:
                # To lowercase if only first letter is uppercase (chemical elements already covered above).
                tok = tok.lower()

            if remove_accents:
                tok = self.remove_accent(tok)

            processed.append(tok)

        if make_phrases:
            processed = self.make_phrases(processed, reps=2)

        return processed, mat_list

    def make_phrases(self, sentence, reps=2):
        """Generates phrases from a sentence of words.

        Args:
            sentence: A list of tokens (strings).
            reps: How many times to combine the words.

        Returns:
            A list of strings where the strings in the original list are combined
            to form phrases, separated from each other with an underscore "_".
        """
        while reps > 0:
            sentence = self.phraser[sentence]
            reps -= 1
        return sentence

    def is_number(self, s):
        """Determines if the supplied string is number.

        Args:
            s: The input string.

        Returns:
            True if the supplied string is a number (both . and , are acceptable), False otherwise.
        """
        return self.NR_BASIC.match(s.replace(",", "")) is not None

    @staticmethod
    def is_element(txt):
        """Checks if the string is a chemical symbol.

        Args:
            txt: The input string.

        Returns:
            True if the string is a chemical symbol, e.g. Hg, Fe, V, etc. False otherwise.
        """
        try:
            Element(txt)
            return True
        except ValueError:
            return False

    def is_simple_formula(self, text):
        """Determines if the string is a simple chemical formula.

        Excludes some roman numbers, e.g. IV.

        Args:
            text: The input string.

        Returns:
            True if the supplied string a simple formula, e.g. IrMn, LiFePO4, etc. More complex
            formula such as LiFePxO4-x are not considered to be simple formulae.
        """
        if self.VALENCE_INFO.search(text) is not None:
            # 2 consecutive II, IV or VI should not be parsed as formula.
            # Related to valence state, so don"t want to mix with I and V elements.
            return False
        elif any(char.isdigit() or char.islower() for char in text):
            # Aas to contain at least one lowercase letter or at least one number (to ignore abbreviations).
            # Also ignores some materials like BN, but these are few and usually written in the same way,
            # so normalization won"t be crucial.
            try:
                if text in ["O2", "N2", "Cl2", "F2", "H2"]:
                    # Including chemical elements that are diatomic at room temperature and atm pressure,
                    # despite them having only a single element.
                    return True
                composition = Composition(text)
                # Has to contain more than one element, single elements are handled differently.
                if len(composition.keys()) < 2 or any([not self.is_element(key) for key in composition.keys()]):
                    return False
                return True
            except (CompositionError, ValueError):
                return False
        else:
            return False

    @staticmethod
    def get_ordered_integer_formula(el_amt, max_denominator=1000):
        """Converts a mapping of {element: stoichiometric value} to a alphabetically ordered string.

        Given a dictionary of {element : stoichiometric value, ..}, returns a string with
        elements ordered alphabetically and stoichiometric values normalized to smallest common
        integer denominator.

        Args:
            el_amt: {element: stoichiometric value} mapping.
            max_denominator: The maximum common denominator of stoichiometric values to use for
                normalization. Smaller stoichiometric fractions will be converted to the same
                integer stoichiometry.

        Returns:
            A material formula string with elements ordered alphabetically and the stoichiometry
            normalized to the smallest integer fractions.
        """
        g = gcd_float(list(el_amt.values()), 1 / max_denominator)
        d = {k: round(v / g) for k, v in el_amt.items()}
        formula = ""
        for k in sorted(d):
            if d[k] > 1:
                formula += k + str(d[k])
            elif d[k] != 0:
                formula += k
        return formula

    def normalized_formula(self, formula, max_denominator=1000):
        """Normalizes chemical formula to smallest common integer denominator, and orders elements alphabetically.

        Args:
            formula: the string formula.
            max_denominator: highest precision for the denominator (1000 by default).

        Returns:
            A normalized formula string, e.g. Ni0.5Fe0.5 -> FeNi.
        """
        try:
            formula_dict = Composition(formula).get_el_amt_dict()
            return self.get_ordered_integer_formula(formula_dict, max_denominator)
        except (CompositionError, ValueError):
            return formula

    @staticmethod
    def remove_accent(txt):
        """Removes accents from a string.

        Args:
            txt: The input string.

        Returns:
            The de-accented string.
        """
        # There is a problem with angstrom sometimes, so ignoring length 1 strings.
        return unidecode.unidecode(txt) if len(txt) > 1 else txt
