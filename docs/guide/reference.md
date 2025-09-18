---
title: Project Reference — Giovanni Castellazzi
hide:
  - toc
---

# Project Group

**Principal Investigator:** Giovanni Castellazzi  
**Affiliations:** Associate Professor in Solids and Structural Mechanics, DICAM — University of Bologna  
**Email:** <giovanni.castellazzi@unibo.it>


| **Collaborators**                  | **Affiliations**                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------|
| [Giovanni Castellazzi](https://www.unibo.it/sitoweb/giovanni.castellazzi/en) | Solids and Structural Mechanics, DICAM — University of Bologna |
| [Stefano de Miranda](https://www.unibo.it/sitoweb/stefano.demiranda/en) | Solids and Structural Mechanics, DICAM — University of Bologna |
| [Antonio Maria D'Altri](https://www.unibo.it/sitoweb/am.daltri/en) | Solids and Structural Mechanics, DICAM — University of Bologna |
| [Francesco Ubertini](https://www.unibo.it/sitoweb/francesco.ubertini/en) | Solids and Structural Mechanics, DICAM — University of Bologna |
| [Nicolò Lo Presti](https://www.unibo.it/sitoweb/nicolo.lopresti2/en) | PhD Student, DICAM — University of Bologna |
| Kaj Kolodziej             | Master Student, University of Bologna                                                              |
| **Former Collaborators**      |                                                                                                     |
| [Gabriele Bitelli](https://www.unibo.it/sitoweb/gabriele.bitelli/en) | Geomatics, DICAM — University of Bologna                                           |
| [Alessandro Lambertini](https://www.unibo.it/sitoweb/alessandro.lambertini/en) | Geomatics, DICAM — University of Bologna                                                  |



---

## Report a Problem / Bug

**Preferred channel:** GitHub Issues (recommended for tracking & transparency)  
**Alternative:** Email support (for private data or access‑restricted material)

### 1) Via GitHub Issues
### Recommended Method for Reporting Problems

The preferred way to report issues is via **GitHub Issues**, as it ensures transparency and allows for better tracking and collaboration. Follow the steps below:

1. Navigate to the repository’s **Issues** page.
2. Click **New issue** and select **Bug report**.
3. Fill out the template, providing as much detail as possible (refer to the checklist below).
4. Attach relevant files, such as screenshots, logs, or minimal datasets. If your data is sensitive, consider using the email option instead.

By using GitHub Issues, you help streamline the process and contribute to improving the project for everyone.

> If your dataset is confidential, please **do not upload it publicly**. Instead, strip/obfuscate sensitive parts or use the email channel.

### 2) Via Email 
In the case of **private** or **urgent** communications send an email to **<giovanni.castellazzi@unibo.it>** with subject:  
`[C2F4DT] Bug report: <short description>`

Include the same checklist information below. You may attach files or share a secure link (e.g., institutional cloud drive).

---

## Bug Report Checklist

Please provide the following to help us reproduce and fix the issue quickly:

1. **Summary**
   - Short, descriptive title.
   - What did you expect to happen? What happened instead?

2. **Steps to Reproduce**
   - Step-by-step instructions.
   - Minimal files (e.g., a small VTK demonstrating the issue).

3. **Screenshots / Videos**
   - UI state, error dialogs, wrong rendering outcomes, etc.

4. **Logs & Console Output**
   - Copy any text from the built-in **Console** or terminal.
   - If available, attach `_vtk_report/report.json` or similar validator outputs.

5. **Environment**
   - OS (Windows/macOS/Linux + version), Python version.
   - Package versions: `pyvista`, `vtk`, `numpy`, `matplotlib`, `scipy`.
   - GPU/driver info if rendering‑related.

6. **Configuration**
   - Representation settings (Surface/Wireframe/etc.).
   - Color by (Solid/PointData/CellData + vector component).
   - LUT + invert, scalar bar on/off, edges visibility, opacity.
   - Any custom plugins or local modifications.

7. **Frequency / Regression**
   - Does it happen always or intermittently?
   - Did it appear after an update? Which version worked?

---

## Example (Minimal) Issue Template

```markdown
**Summary**
Surface with Edges hides nodes when edges are enabled.

**Steps to Reproduce**
1. File ▸ Import VTK… → load `demo.vtu`
2. Representation = Surface with Edges
3. Color by = PointData/displacement (Magnitude)
4. Toggle Edges = ON

**Observed**
Edges are visible but nodes are white.

**Expected**
Edges and scalar-colored nodes visible together.

**Environment**
- OS: Windows 11, Python 3.11
- pyvista 0.43.10, vtk 9.3.0, numpy 2.0.2, matplotlib 3.9.2, scipy 1.13
- GPU: NVIDIA RTX 3070 (driver 552.xx)

**Attachments**
- `_vtk_report/rep_Surface_with_Edges.png`
- Console log excerpt
```

---

## Optional: GitHub Issue Templates

To streamline reports, consider adding `.github/ISSUE_TEMPLATE/` files to your repo:

- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`

**Example `bug_report.md`:**

```yaml
name: Bug report
description: Create a report to help us improve
labels: [bug]
body:
  - type: textarea
    id: summary
    attributes:
      label: Summary
      description: What happened and what did you expect?
      placeholder: Short description...
    validations:
      required: true

  - type: textarea
    id: repro
    attributes:
      label: Steps to Reproduce
      description: Provide a minimal, complete, and verifiable example.
      placeholder: |
        1. Import file ...
        2. Set representation to ...
        3. Toggle edges ...
    validations:
      required: true

  - type: textarea
    id: evidence
    attributes:
      label: Evidence
      description: Screenshots, logs, minimal datasets, validator output.
    validations:
      required: false

  - type: input
    id: env-os
    attributes:
      label: OS
      placeholder: Windows 11 / macOS 14 / Ubuntu 22.04
    validations:
      required: true

  - type: input
    id: env-python
    attributes:
      label: Python
      placeholder: 3.11.x
    validations:
      required: true

  - type: textarea
    id: env-packages
    attributes:
      label: Package Versions
      placeholder: pyvista, vtk, numpy, matplotlib, scipy
    validations:
      required: true
```

---

## FAQ

**Q: I found a bug while using a confidential dataset. How should I proceed?**  
A: Prefer the **email channel**. Provide a minimal, anonymized extract if possible, or share access privately.

**Q: Where do I ask for new features or improvements?**  
A: Use **GitHub Issues → Feature request** to track discussion and prioritization.

**Q: I’m offline and cannot reach GitHub.**  
A: Send an email to <giovanni.castellazzi@unibo.it> with the **Bug Report Checklist** filled in.

---

*Thank you for helping improve C2F4DT! Your clear reports save everyone time and make the tool better for the community.*
