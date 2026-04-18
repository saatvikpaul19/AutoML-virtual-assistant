import os; os.environ['TRANSFORMERS_NO_TF']='1'
from modules.C_nlu.nlu_pipeline import understand

tests = [
    ('hello', 'greetings'),
    ('thank you', 'thanks'),
    ('sleep', 'sleep_va'),
    ('can you do my groceries', 'out_of_scope'),
    ('what is the weather in Ottawa today', 'get_weather'),
    ('set a timer for 5 minutes', 'set_timer'),
    ('load iris dataset', 'load_dataset'),
    ('set learning rate to 0.001', 'set_learning_rate'),
]

print('\n=== NLU SMOKE TESTS ===\n')
passed = 0
for text, expected in tests:
    r = understand(text)
    ok = r['intent'] == expected
    status = 'PASS' if ok else f'FAIL (got {r["intent"]})'
    print(f'[{status}]  "{text}"')
    print(f'         slots={r["slots"]}  fallback={r["fallback_used"]}')
    if r.get('fallback_reason'):
        print(f'         reason={r["fallback_reason"][:80]}')
    print()
    if ok: passed += 1

print(f'=== {passed}/{len(tests)} passed ===')
