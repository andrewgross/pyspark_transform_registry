import re


def _filter_requirements(
    requirements_path: str, exclude_packages: list[str]
) -> list[str]:
    """
    Filter requirements to exclude specified packages.

    Args:
        requirements_path: Path to the requirements file
        exclude_packages: List of packages to exclude

    Returns:
        List of package names
    """
    filtered_packages = []
    package_names = _extract_package_names(requirements_path)
    for normalized_package_name, original_requirement in package_names:
        if not any(
            _normalize_package_name(exclude_package) == normalized_package_name
            for exclude_package in exclude_packages
        ):
            filtered_packages.append(original_requirement)
    return filtered_packages


def _extract_package_names(requirements_file: str) -> list[tuple[str, str]]:
    """
    Extract package names from a requirements file.

    Returns only the package names, removes the version, extras, and other metadata.

    Args:
        requirements_file: Path to the requirements file

    Returns:
        List of package names
    """
    package_names: list[str] = []
    with open(requirements_file, "r") as f:
        for line in f.readlines():
            line = line.strip()
            original_requirement = line
            if not line or line.startswith("#"):
                continue

            # Remove environment markers (e.g., "; python_version >= '3.6'")
            line = line.split(";", 1)[0].strip()

            # Handle editable installs (-e/--editable)
            if line.startswith(("-e", "--editable")):
                # Extract URL part
                if line.startswith("--editable="):
                    url = line[len("--editable=") :]
                else:
                    tokens = line.split()
                    url = " ".join(tokens[1:]) if len(tokens) > 1 else ""

                # Extract package from #egg= fragment
                match = re.search(r"#egg=([^&\s]+)", url)
                if match:
                    package_names.append(
                        (_normalize_package_name(match.group(1)), original_requirement)
                    )
                continue

            # Extract package name from standard requirement
            match = re.match(r"^([a-zA-Z_][a-zA-Z0-9._-]*)", line)
            if match:
                package_names.append(
                    (_normalize_package_name(match.group(1)), original_requirement)
                )

    return package_names


def _normalize_package_name(package_name: str) -> str:
    """
    Normalize a package name to lowercase and replace underscores with dashes.
    """
    return package_name.strip().replace("_", "-").lower()
