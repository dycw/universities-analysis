from pytest import mark

from universities_analysis.data import BST_DESTINATIONS, LFIT_DESTINATIONS
from universities_analysis.read import get_metric, read_data


def test_read_data() -> None:
    df = read_data()
    dtypes = {
        "NationalRank": float,
        "RegionalRank": float,
        "Rank2022": float,
        "Rank2021": float,
        "InstitutionName": "string",
        "LocationCode": "string",
        "LocationCountry": "string",
        "ClassificationSize": "string",
        "ClassificationFocus": "string",
        "ClassificationRes": "string",
        "ClassificationAge": int,
        "ClassificationStatus": "string",
        "AcademicReputatationScore": float,
        "AcademicReputatationRank": float,
        "EmployerReputationScore": float,
        "EmployerReputationRank": float,
        "FacultyStudentScore": float,
        "FacultyStudentRank": float,
        "CitationsPerFacultyScore": float,
        "CitationsPerFacultyRank": float,
        "InternationalFacultyScore": float,
        "InternationalFacultyRank": float,
        "InternationalStudentsScore": float,
        "InternationalStudentsRank": float,
        "OverallScore": float,
    }
    assert dict(df.dtypes) == dtypes
    assert df.shape == (1300, 25)


@mark.parametrize("name", BST_DESTINATIONS + LFIT_DESTINATIONS)
def test_get_overall_score(name: str) -> None:
    _ = get_metric(name)
