# Inferox RFCs

This directory contains Request for Comments (RFC) documents for substantial changes to Inferox.

## What is an RFC?

An RFC is a design document that describes a new feature, architectural change, or significant modification to Inferox. The RFC process allows the community to discuss and refine proposals before implementation begins.

## When to write an RFC

You should write an RFC for:

- **New features** that affect the public API
- **Breaking changes** to existing functionality
- **Architectural changes** that affect multiple crates
- **New file formats** or protocols
- **Major performance optimizations** that change behavior

You probably don't need an RFC for:

- Bug fixes
- Documentation improvements
- Internal refactoring (unless it affects the public API)
- Minor performance improvements
- Adding tests

## RFC Process

1. **Draft**: Copy `0000-template.md` and write your proposal
2. **Discussion**: Open a PR with your RFC for community feedback
3. **Revision**: Update based on feedback
4. **Acceptance**: Core team reviews and accepts/rejects
5. **Implementation**: Implement the RFC (can happen in parallel with discussion)

## RFC Lifecycle

- **Draft**: Work in progress, not yet ready for review
- **Proposed**: Ready for community review (PR open)
- **Accepted**: Approved for implementation
- **Implemented**: Feature is merged into main
- **Rejected**: Not accepted (with rationale)
- **Superseded**: Replaced by a newer RFC

## Active RFCs

- [RFC 0001: Model Distribution Format (.ilmdl)](./0001-model-distribution-format.md) - **Draft**

## Implemented RFCs

None yet.

## Template

See `0000-template.md` for the RFC template.
