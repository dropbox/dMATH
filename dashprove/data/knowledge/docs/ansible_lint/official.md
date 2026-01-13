# Ansible Lint Documentation[¶][1]

## About Ansible Lint[¶][2]

Ansible Lint is a command-line tool for linting **playbooks, roles and collections** aimed toward
any Ansible users. Its main goal is to promote proven practices, patterns and behaviors while
avoiding common pitfalls that can easily lead to bugs or make code harder to maintain.

Ansible lint is also supposed to help users upgrade their code to work with newer versions of
Ansible. Due to this reason we recommend using it with the newest version of Ansible, even if the
version used in production may be older.

As any other linter, it is opinionated. Still, its rules are the result of community contributions
and they can always be disabled based individually or by category by each user.

[Ansible Galaxy project][3] makes use of this linter to compute quality scores for [Galaxy Hub][4]
contributed content. This does not mean this tool only targets those that want to share their code.
Files like `galaxy.yml`, or sections like `galaxy_info` inside `meta.yml` help with documentation
and maintenance, even for unpublished roles or collections.

The project was originally started by [@willthames][5] and has since been adopted by the Ansible
Community team. Its development is purely community driven while keeping permanent communications
with other Ansible teams.

[1]: #ansible-lint-documentation
[2]: #about-ansible-lint
[3]: https://github.com/ansible/galaxy/
[4]: https://galaxy.ansible.com/
[5]: https://github.com/willthames/
