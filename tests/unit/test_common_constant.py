from uniparser_tools.common.constant import PPOCR_LANG, SUPPORTED_LANG, Language, to_ocr_lang


def test_ppocr_language_mapping_matches_parser_models() -> None:
    assert PPOCR_LANG[Language.German] == "de"
    assert PPOCR_LANG[Language.Serbian] == "rs_cyrillic"
    assert PPOCR_LANG[Language.Serbian_Latin] == "rs_latin"
    assert Language.Persian in SUPPORTED_LANG


def test_to_ocr_lang_handles_aliases_and_unknown_values() -> None:
    assert to_ocr_lang(Language.Cantonese) == Language.Chinese_Traditional
    assert to_ocr_lang(Language.Western_Mari) == Language.Eastern_Mari
    assert to_ocr_lang("not-a-language") == Language.Unknown
