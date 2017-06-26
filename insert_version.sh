#!/bin/sh
#
# Insert (if called as "insert_version.sh") / remove (if called "del_version.sh")
# version tag in python file. Use it as smudge / clean filter
#
#
# put it in your $PATH as "insert_version.sh" and do "ln -s insert_version.sh del_version.sh"
#

script_name="${0##*/}"

insert="insert_version.sh"


if [ "$script_name" = "$insert" ]; then
        version_nr=`git describe --tag --abbrev=6 --dirty --always`
        version_post=`git log -1 --date=short --format=\(%ad\)\ released\ by\ \<%an\>\ %ae`

        version="${version_nr} ${version_post}"
        sed  "s/^__version__ = ''/__version__ = '${version}'/" $*
else
        sed  "s/^__version__ = '.*'/__version__ = ''/" $*
fi


