# Security

This is research code. It is not a supported product, and it gets no security updates.

To report a security problem, use GitHub's private vulnerability reporting: the **Security** tab of this
repository, then **Report a vulnerability**. Please do not open a public issue for it.

Two parts run code that is not ours:
- `eval/lcb_sandbox.py` runs model-written programs. It does this only inside a container with no network.
  Do not run `eval/lcb_sandbox_runner.py` on a host directly.
- The SGLang patch loads kernels from the path in `LEAPQUANT_HOME`. Point it only at a checkout you trust.
