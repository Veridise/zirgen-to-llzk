# Contribution Guide {#contribution-guide}

\note This contribution guide is adapted from the example guide provided [by GitHub][git-example-guide].
We will continue to refine the information here as needed.

\tableofcontents

Thank you for investing your time in contributing to the Zklang project!

Before getting involved, read our \subpage code-of-conduct "Code of Conduct"
to keep our community approachable and respectable.

In this guide you will get an overview of the contribution workflow from
opening an issue, creating a PR, reviewing, and merging the PR.

## Getting started

### Issues

#### Create a new issue

If you spot a problem, encounter a bug, or want to request a new feature,
[search if an issue already exists][issues-page].
If a related issue doesn't exist, you can open a new issue using a relevant [issue form][new-issue].

#### Solve an issue

Scan through our [existing issues][issues-page] to find one that interests you.
You can narrow down the search using `labels` as filters.
See ["Label reference"][git-labels] for more information.
As a general rule, we don’t assign issues to anyone.
If you find an issue to work on, you are welcome to open a PR with a fix.

### Make Changes

1. Fork the repository.
- Using GitHub Desktop:
  - [Getting started with GitHub Desktop][git-desktop] will guide you through setting up Desktop.
  - Once Desktop is set up, you can use it to [fork the repo][git-desktop-fork]!

- Using the command line:
  - [Fork the repo][git-fork] so that you can make your changes without affecting the original project until you're ready to merge them.

2. Install or update LLZK development dependencies. For more information, see \ref setup "The Development Guide".

3. Create a working branch and start with your changes!

### Commit your update

Commit the changes once you are happy with them.
Remember that all code must pass a format check before it can be merged, so consider running clang format
before pushing your commits. For more information, see \ref dev-workflow "the development workflow section".

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.
- All PRs require a changelog describing what user-level changes have been made in the PR. To create a template changelog, run `create-changelog` from the nix shell.
- Don't forget to [link PR to issue][git-link-issue] if you are solving one.
- Enable the checkbox to [allow maintainer edits][git-maintainer-edits] so the branch can be updated for a merge.
Once you submit your PR, a maintainer will review your proposal. We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes][git-feedback] or pull request comments.
You can apply suggested changes directly through the UI.
You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved][git-resolving-conversations].
- If you run into any merge issues, checkout this [git tutorial][git-merge-conflict-tutorial] to help you resolve merge conflicts and other issues.

### Your PR is merged!

Congratulations! The Zklang team and LLZK community thanks you.

Once your PR is merged, your contributions will be publicly available in the [Zklang repository][zklang-repo].

[issues-page]: https://github.com/Veridise/zirgen-to-llzk/issues
[new-issue]: https://github.com/Veridise/zirgen-to-llzk/issues/new/choose
[zklang-repo]: https://github.com/Veridise/zirgen-to-llzk

[git-example-guide]: https://github.com/github/docs/blob/278ce65fe7e7cb7e8432e9f032f94c7fe46c379e/.github/CONTRIBUTING.md
[git-labels]: https://docs.github.com/en/contributing/collaborating-on-github-docs/label-reference
[git-fork]: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository
[git-desktop]: https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop
[git-desktop-fork]: https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop
[git-link-issue]: https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
[git-maintainer-edits]: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork
[git-feedback]: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request
[git-resolving-conversations]: https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations
[git-merge-conflict-tutorial]: https://github.com/skills/resolve-merge-conflicts

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref tools | \ref dialects |
</div>
