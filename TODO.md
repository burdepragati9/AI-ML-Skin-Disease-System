# AI Out-of-Scope Prediction Fix Implementation

## Steps

- [x] Create TODO.md
- [x] FIX 1: `utils/prediction_comparison.py` — Validate AI override in `build_ai_verification_summary()`
- [x] FIX 2: `utils/ai_recognition.py` — Constrain Gemini prompts to 4 active classes
- [x] FIX 3: `backend/services/prediction_service.py` — Final prediction safety validation gate
- [x] FIX 4: `history/search_history.py` — History/database safety validation
- [x] FIX 5: `training/self_learning.py` — Self-learning safety to prevent expanding 4-class scope
- [ ] Test all 7 scenarios

