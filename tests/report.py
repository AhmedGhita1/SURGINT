class TestReport:
    def __init__(self, module_name):
        self.module_name = module_name
        self.passed = 0
        self.failed = 0

    def run(self, name, test):
        try:
            test()
        except Exception as exc:
            self.failed += 1
            print(f"  ❌ {name}: {str(exc).strip() or type(exc).__name__}")
        else:
            self.passed += 1
            print(f"  ✅ {name}")

    def finish(self):
        print("\n" + "=" * 50)
        print(f"{self.module_name} test summary")
        print(f"  ✅ Passed: {self.passed}")
        print(f"  ❌ Failed: {self.failed}")
        print(f"  📊 Total:  {self.passed + self.failed}")
        print("=" * 50)
        return self.failed == 0
