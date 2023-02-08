from contextlib import suppress
from functools import cache, partial

from beartype import beartype
from numpy import nan
from pandas import DataFrame, Series, read_excel
from utilities.re import NoMatchesError, extract_group, extract_groups

from universities_analysis.data import BST_DESTINATIONS, LFIT_DESTINATIONS


@cache
@beartype
def read_data() -> DataFrame:
    """Read the QS World University Rankings Results."""
    names = [
        "NationalRank",
        "RegionalRank",
        "Rank2022",
        "Rank2021",
        "InstitutionName",
        "LocationCode",
        "LocationCountry",
        "ClassificationSize",
        "ClassificationFocus",
        "ClassificationRes",
        "ClassificationAge",
        "ClassificationStatus",
        "AcademicReputatationScore",
        "AcademicReputatationRank",
        "EmployerReputationScore",
        "EmployerReputationRank",
        "FacultyStudentScore",
        "FacultyStudentRank",
        "CitationsPerFacultyScore",
        "CitationsPerFacultyRank",
        "InternationalFacultyScore",
        "InternationalFacultyRank",
        "InternationalStudentsScore",
        "InternationalStudentsRank",
        "OverallScore",
    ]
    rank_columns = {
        "NationalRank",
        "RegionalRank",
        "Rank2022",
        "Rank2021",
        "AcademicReputatationRank",
        "EmployerReputationRank",
        "FacultyStudentRank",
        "CitationsPerFacultyRank",
        "InternationalFacultyRank",
        "InternationalStudentsRank",
        "OverallScore",
    }
    return read_excel(
        "src/assets/2022_QS_World_University_Rankings_Results_public_version.xlsx",
        converters={col: _convert_rank for col in rank_columns}
        | {"OverallScore": _convert_overall_score},
        dtype={
            "InstitutionName": "string",
            "LocationCode": "string",
            "LocationCountry": "string",
            "ClassificationSize": "string",
            "ClassificationFocus": "string",
            "ClassificationRes": "string",
            "ClassificationStatus": "string",
        },
        header=None,
        names=names,
        skiprows=4,
    ).astype({"Rank2021": float})


@beartype
def _convert_rank(rank: int | str, /) -> float:
    if isinstance(rank, int):
        return float(rank)
    if rank == "":
        return nan
    with suppress(NoMatchesError):
        return float(extract_group(r"^\s*(\d+)[\s+=]*$", rank))
    low, high = map(float, extract_groups(r"^\s*(\d+)-(\d+)\s*$", rank))
    return (low + high) / 2


@beartype
def _convert_overall_score(score: int | float | str, /) -> float:
    if isinstance(score, (int, float)):
        return float(score)
    _ = extract_group(r"^(\-)$", score)
    return nan


_ABSENT = {
    "Bocconi School of Management",
    "CPES Paris Sciences & Lettres + Lycée Henri-IV",
    "California College of the Arts",
    "EDHEC Business School",
    "EM Lyon",
    "ENSA School of Architecture",
    "ESCP Europe",
    "ESSEC School of Management",
    "Ecole du Louvre",
    "Ecole hôtelière de Lausanne",
    "HTW Berlin",
    "Hospitality Management School of The Hague",
    "IESEG School of Management",
    "INSA Lyon",
    "ISIPCA Paris",
    "ISTR Lyon",
    "Institut d'Administration des Entreprises",
    "LUISS Guido Carli",
    "Liverpool Hope University",
    "Maastricht University",
    "Middlebury College",
    "Musashino Art University",
    "München Universität",
    "Olivier de Serres",
    "Penninghen Institute of Art and Design",
    "United International Business Schools",
    "University of California",
    "University of Global Business",
    "University of Mechelen",
    "University of Toronto Scarborough",
    "University of the Arts London",
    "Université Libre",
    "Université René Descartes -Paris-V",
    "Universités de Technologie",
    "Vatel International Hospitality Management School",
}
_RENAME = {
    "Bachelor of the Ecole Polytechnique": "Institut Polytechnique de Paris",
    "Cambridge University": "University of Cambridge",
    "Ecole Polytechnique Fédérale de Lausanne": "University of Lausanne",
    "Eidgenössische Technische Hochschule Zürich": "ETH Zurich - Swiss Federal Institute of Technology",  # noqa: E501
    "Ferrandi International Hospitality Management School": "Universita' degli Studi di Ferrara",  # noqa: E501
    "Imperial College": "Imperial College London",
    "Kingston University London": "Kingston University, London",
    "London School of Economics": "The London School of Economics and Political Science (LSE)",  # noqa: E501
    "Mc Gill University": "McGill University",
    "Sciences-Po Paris": "Sciences Po ",
    "The University of British Columbia": "University of British Columbia",
    "University College London": "UCL",
    "University College of London": "UCL",
    "University of Gröningen": "University of Groningen",
    "University of Rotterdam": "Erasmus University Rotterdam ",
    "Università Cattolica del Sacro Cuore": "Università\xa0Cattolica del Sacro Cuore",  # noqa: E501
    "Université d'Assas -Paris-II": "University Paris 2 Panthéon-Assas",
    "Université de Louvain": "Université catholique de Louvain (UCLouvain)",
    "Université de Montréal": "Université de Montréal ",
    "Universités de Paris La Sorbonne": "Sorbonne University",
    "Warwick University": "The University of Warwick",
}


@beartype
def get_metric(name: str, /, *, metric: str = "OverallScore") -> float:
    """Get the metric for a university."""
    df = read_data()
    try:
        return df.loc[df["InstitutionName"] == name, metric].item()
    except ValueError:
        if name in _ABSENT:
            return nan
        with suppress(KeyError):
            return get_metric(_RENAME[name], metric=metric)
        raise


@beartype
def get_dataframe_of_scores(names: list[str], /) -> DataFrame:
    """Get the overall scores for a list of names."""
    df = Series(names, name="Name").to_frame()
    df["Rank2022"] = df.Name.map(partial(get_metric, metric="Rank2022"))
    df["Score"] = df.Name.map(get_metric)
    return df.sort_values("Rank2022").reset_index(drop=True)


BST_SCORES, LFIT_SCORES = map(
    get_dataframe_of_scores,
    [BST_DESTINATIONS, LFIT_DESTINATIONS],
)
