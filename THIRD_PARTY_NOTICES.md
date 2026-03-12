# Third-Party Notices

This repository includes code derived from the following project:

## LenslessPiCam
- Upstream repository: https://github.com/LCAV/LenslessPiCam
- License: GNU Affero General Public License v3.0 (AGPL-3.0)

### Derived code in this repository
The file below includes code copied and modified from LenslessPiCam:
- `lensless_flow/data.py`

### Upstream source files referenced
Portions were derived from files including:
- `lensless/utils/dataset.py`
- `lensless/utils/io.py`

### Modifications made in this repository
Modifications include, at least:
- extracting only the dataset loading-related pieces needed by this project;
- adapting the code to run in this repository structure;
- making compatibility changes for Python 3.12;
- removing unrelated functionality and dependencies not needed here.

See the Git history for exact changes and dates.