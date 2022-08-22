#!/bin/bash

VERSION=$(cat setup.py| grep version | cut -c 20- | rev | cut -c 3- | rev)
COMMIT_MESSAGE="$(git log  --format=%B -n 1)"

git config --global user.email "diviyan@circleci.com"
git config --global user.name "CircleCI Bumpversion"
git config --global push.default simple
pip install bumpversion
if [[ $COMMIT_MESSAGE == *"[NV]"* ]] || [[ $COMMIT_MESSAGE == *"[DOC]"* ]];
    then echo "No version update";
    elif [[ $COMMIT_MESSAGE == *"[REL]"* ]];
    then bumpversion --current-version $VERSION minor setup.py README.md docs/index.rst docs/conf.py cdt/__init__.py;
    elif [[ $COMMIT_MESSAGE == *"[MREL]"* ]];
    then bumpversion --current-version $VERSION major setup.py README.md docs/index.rst docs/conf.py cdt/__init__.py;
    else bumpversion --current-version $VERSION patch setup.py README.md docs/index.rst docs/conf.py cdt/__init__.py;
fi
