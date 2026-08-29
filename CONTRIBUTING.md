# Contributing Guide

## Branch Rules

Do not develop directly on the main branch.

Each team member should work on their own feature branch.

Zhang Yiran
Branch: feature/inference
Responsibility: Inference pipeline and prediction script

Liu Zhenning
Branch: feature/robustness-evaluation
Responsibility: Robustness evaluation

Chen Yizhen
Branch: feature/repository-management
Responsibility: Repository structure and project integration

Zhou Shuyuan
Branch: feature/error-analysis
Responsibility: Error analysis and explainability

Wang Zihan
Branch: feature/devpost-demo
Responsibility: Devpost documentation and demo video

The workflow is:

main
↓
feature branch
↓
development
↓
commit
↓
push
↓
Pull Request
↓
review
↓
merge into main


## Commit Rules

Use short and clear commit messages.

Recommended format:

type: short description

Examples:

feat: add prediction script
feat: add image transformation pipeline
fix: handle corrupted image files
docs: update README
docs: add error analysis report
refactor: reorganise evaluation code

Recommended commit types:

feat = add a new feature
fix = fix a bug
docs = update documentation
refactor = reorganise code without changing its main functionality
test = add or update tests
chore = maintenance or configuration changes

Avoid unclear commit messages such as:

update
change
final
test
123

Each commit should represent one clear and meaningful change.


## File Organisation

Keep files in the appropriate directories.

robust-aigc-detector/
├── analysis/
├── assets/
├── configs/
├── evaluation/
├── scripts/
├── src/
├── .gitignore
├── CONTRIBUTING.md
├── README.md
└── requirements.txt

Directory guidelines:

src/ = model and core source code

scripts/ = executable scripts such as inference and data-processing scripts

evaluation/ = evaluation code and robustness evaluation results

analysis/ = error analysis, visualisations, and related reports

configs/ = configuration files

assets/ = project assets, figures, and demonstration materials

README.md = main project documentation

CONTRIBUTING.md = Git and collaboration guidelines

requirements.txt = Python dependencies

Do not commit large datasets or model weights unless explicitly required and permitted.


## Pull Request Rules

Do not merge directly into main without checking the changes.

Before opening a Pull Request:

1. Make sure the code runs locally.
2. Make sure the changes are limited to the intended task.
3. Make sure files are placed in the correct directories.
4. Commit the changes with a clear commit message.
5. Push the feature branch to GitHub.

Pull Requests should include:

- A short description of the changes.
- The reason for the changes.
- Any important testing or evaluation results.
- Any known issues or limitations.


## Collaboration Workflow

The team follows this workflow:

Create branch
↓
Develop and test
↓
git add
↓
git commit
↓
git push
↓
Open Pull Request
↓
Review
↓
Merge into main

The main branch should contain the latest stable version of the project.


## Project Integration

The final repository should integrate:

- Inference and prediction pipeline from Member A.
- Robustness evaluation pipeline and results from Member B.
- Repository documentation and project management from Member C.
- Error analysis and explainability results from Member D.
- Devpost documentation and demo materials from Member E.

The upstream MIRROR project license and attribution should be preserved in the repository.