# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

FEATURES_LIST_PLACEHOLDER = "[[main-features-list]]"
HOME_PAGE_TEMPLATE_RELATIVE_PATH = "README.template.md"  # relative to doc dir
HOME_PAGE_RELATIVE_PATH = "README.md"  # relative to doc dir
REPO_README_RELATIVE_PATH = "README.md"  # relative to root dir
FEATURES_LIST_START_TAG = "<!--features-list-start-->"
FEATURES_LIST_END_TAG = "<!--features-list-end-->"


def extract_list_features(rootdir: str) -> str:
    with open(f"{rootdir}/{REPO_README_RELATIVE_PATH}", "rt") as f:
        repo_readme_text = f.read()

    start = repo_readme_text.find(FEATURES_LIST_START_TAG) + len(
        FEATURES_LIST_START_TAG
    )
    end = repo_readme_text.find(FEATURES_LIST_END_TAG)
    if start > -1 and end > -1:
        features_list_text = repo_readme_text[start:end]
    else:
        features_list_text = ""

    return features_list_text


def create_homepage_with_features_list(rootdir: str, docdir: str) -> None:
    features_list_text = extract_list_features(rootdir=rootdir)

    with open(f"{docdir}/{HOME_PAGE_TEMPLATE_RELATIVE_PATH}", "rt") as f:
        readme_template_text = f.read()

    readme_text = readme_template_text.replace(
        FEATURES_LIST_PLACEHOLDER, features_list_text
    )

    with open(f"{docdir}/{HOME_PAGE_RELATIVE_PATH}", "wt") as f:
        f.write(readme_text)


if __name__ == "__main__":
    docdir = os.path.dirname(os.path.abspath(__file__))
    rootdir = os.path.abspath(f"{docdir}/..")

    create_homepage_with_features_list(rootdir=rootdir, docdir=docdir)
