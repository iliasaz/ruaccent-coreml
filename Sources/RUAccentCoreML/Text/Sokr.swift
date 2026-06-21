import Foundation

/// Abbreviation / initial lists consulted by the razdel sentenize rules — transcribed
/// **verbatim** from `razdel/segmenters/sokr.py` (§2.3). Build as `Set<String>` /
/// `Set<[String]>` (pairs). All entries are lowercase as written.
///
/// The `жен р`/`муж р` dedup bug (a missing trailing comma fused them into a dead 3-tuple)
/// is preserved by OMISSION — neither `("жен","р")` nor `("муж","р")` is present (§2.3).
enum Sokr {

    // MARK: single-word sokrs

    /// `TAIL_SOKRS` (sokr.py:14).
    private static let tailSokrs: [String] = """
    дес тыс млн млрд дол долл коп руб р проц га барр куб кв км см
    час мин сек в вв г гг с стр co corp inc изд ed др al
    """.split(whereSeparator: { $0 == " " || $0 == "\n" }).map(String.init)

    /// `HEAD_SOKRS` (sokr.py:40).
    static let headSokrs: Set<String> = Set("""
    букв ст трад
    лат венг исп кат укр нем англ фр итал греч
    евр араб яп слав кит рус русск латв словацк хорв
    mr mrs ms dr vs св арх зав зам проф акад кн корр ред гр ср
    чл корр им тов нач пол chap
    п пп ст ч чч гл стр абз пт no
    просп пр ул ш г гор д стр к корп пер корп обл эт пом ауд оф ком комн каб
    домовлад лит т рп пос с х пл bd о оз р а
    обр ум ок откр пс ps upd см напр доп юр физ тел сб внутр дифф гос отм
    """.split(whereSeparator: { $0 == " " || $0 == "\n" }).map(String.init))

    /// `OTHER_SOKRS` (sokr.py:97).
    private static let otherSokrs: [String] = """
    сокр рис искл прим яз устар шутл
    """.split(whereSeparator: { $0 == " " || $0 == "\n" }).map(String.init)

    /// `SOKRS = TAIL_SOKRS | HEAD_SOKRS | OTHER_SOKRS` (sokr.py:105).
    static let sokrs: Set<String> = Set(tailSokrs).union(headSokrs).union(otherSokrs)

    // MARK: pair sokrs

    /// `TAIL_PAIR_SOKRS` (sokr.py:107).
    private static let tailPairSokrs: [[String]] = [
        ["т", "п"], ["т", "д"], ["у", "е"], ["н", "э"], ["p", "m"], ["a", "m"], ["с", "г"], ["р", "х"],
        ["с", "г"], ["с", "ш"], ["з", "д"], ["л", "с"], ["ч", "т"], ["т", "д"],
    ]

    /// `HEAD_PAIR_SOKRS` (sokr.py:123) — also used directly by rule 5.
    static let headPairSokrs: Set<[String]> = Set([
        ["т", "е"], ["т", "к"], ["т", "н"], ["и", "о"], ["к", "н"], ["к", "п"], ["п", "н"], ["к", "т"], ["т", "н"], ["л", "д"],
    ])

    /// `OTHER_PAIR_SOKRS` (sokr.py:134) — the `жен р`/`муж р` 3-tuple is dead, so omitted.
    private static let otherPairSokrs: [[String]] = [
        ["ед", "ч"], ["мн", "ч"], ["повел", "накл"],
    ]

    /// `PAIR_SOKRS = TAIL_PAIR_SOKRS | HEAD_PAIR_SOKRS | OTHER_PAIR_SOKRS` (sokr.py:142).
    static let pairSokrs: Set<[String]> = Set(tailPairSokrs)
        .union(headPairSokrs)
        .union(otherPairSokrs)

    /// `INITIALS` (sokr.py:144).
    static let initials: Set<String> = ["дж", "ed", "вс"]
}
