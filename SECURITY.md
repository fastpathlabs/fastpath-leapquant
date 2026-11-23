# Security

This is research code. It is not a supported product, and it gets no security updates.

To report a security problem, use GitHub's private vulnerability reporting: the **Security** tab of this
repository, then **Report a vulnerability**. If that button is not there, open an issue that says only
"security contact, please", with no details, and we will reply with a private channel. Please do not put
the details of a security problem in a public issue.

Two parts run code that is not ours:
- `eval/lcb_sandbox.py` runs model-written programs. It does this only inside a container with no network.
  Do not run `eval/lcb_sandbox_runner.py` on a host directly.
- The SGLang patch loads kernels from the path in `LEAPQUANT_HOME`. Point it only at a checkout you trust.
