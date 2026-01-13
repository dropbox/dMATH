# Reference Implementations

Clone competing systems here for method reference:

```bash
cd references

# Critical - must understand all methods
git clone --depth 1 https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
git clone --depth 1 https://github.com/oval-group/oval-bab.git
git clone --depth 1 https://github.com/eth-sri/eran.git

# Important alternatives
git clone --depth 1 https://github.com/stanleybak/nnenum.git
git clone --depth 1 https://github.com/NeuralNetworkVerification/Marabou.git

# Additional
git clone --depth 1 https://github.com/verivital/nnv.git
git clone --depth 1 https://github.com/vas-group-imperial/VeriNet.git
git clone --depth 1 https://github.com/Verified-Intelligence/auto_LiRPA.git
```

## Key Files to Study

### α,β-CROWN
- `complete_verifier/bab_attack/bab.py` - Branch and bound
- `complete_verifier/cuts/cut_manager.py` - GCP-CROWN cuts
- `complete_verifier/cuts/biccos.py` - BICCOS
- `auto_LiRPA/bound_ops.py` - CROWN propagation

### OVAL
- `src/branch_and_bound/fsb.py` - FSB branching heuristic
- `src/relaxations/` - Relaxation methods

### ERAN
- `tf_verify/deeppoly/` - DeepPoly implementation
- `ELINA/` - Abstract domains

See `docs/COMPETING-SYSTEMS.md` for full method inventory.
