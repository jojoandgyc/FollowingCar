"""Independent collection: an old main() assertion cannot hide later cases."""
import pytest

import test_identity_bank as legacy

CASES = sorted(name for name in vars(legacy) if name.startswith('_assert_'))


@pytest.mark.parametrize('case', CASES)
def test_legacy_identity_case(case):
    getattr(legacy, case)()
