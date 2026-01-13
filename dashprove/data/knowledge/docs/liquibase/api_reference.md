1. [Documentation Home][1]
3. [Liquibase Pro][2]
5. [User guide, Version 4.33][3]
7. What is a Change type?

## What is a Change type?

A Change Type is a database-independent XML, YAML, or JSON formatted change that you can specify to
update your database with Liquibase. Change Types correspond to SQL statements applied to your
database, such as `CREATE TABLE`. You specify the Change Type you want to use within a
[Changeset][4] in your [Changelog][5]. **It is a best practice to include only one Change Type per
changeset.** Doing so avoids failed auto-commit statements that can leave the database in an
unexpected state.

This page lists all changes that you can apply to your database with the default Liquibase
installation. You can also write your own changes through the extension system.

## Notes

* Property values are *strings *unless otherwise noted.
* Boolean parameters are defaulted to *false *unless otherwise noted.
* Some change types automatically create rollback statements when you run rollback commands.

[1]: /
[2]: /pro
[3]: /pro/user-guide-4-33
[4]: https://docs.liquibase.com/concepts/changelogs/changeset.html
[5]: https://docs.liquibase.com/concepts/changelogs/home.html
