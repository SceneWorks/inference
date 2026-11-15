"%REVIEWED_PYTHON%" scripts/ci/make_test_image.py "%EDIT_SRC%" 768 768 || exit /b 1
"%REVIEWED_PYTHON%" scripts/ci/make_test_image.py "%IP_REF%" 768 768 || exit /b 1
