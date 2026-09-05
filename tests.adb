with Ada.Text_IO; use Ada.Text_IO;
with Almeida_Pineda; use Almeida_Pineda;

procedure Tests is
   Pass_Count : Natural := 0;
   Fail_Count : Natural := 0;

   procedure Check (Label : String; OK : Boolean) is
   begin
      if OK then
         Put_Line ("  PASS — " & Label);
         Pass_Count := Pass_Count + 1;
      else
         Put_Line ("  FAIL — " & Label);
         Fail_Count := Fail_Count + 1;
      end if;
   end Check;

   function Is_Close (A, B : Real) return Boolean is
   begin
      return abs (A - B) < 1.0e-5;
   end Is_Close;

   -- Test Variables
   Net1      : Network (Size => 1);
   Net2      : Network (Size => 2);
   Vec1      : Vector (1 .. 1) := (others => 0.0);
   Vec2      : Vector (1 .. 2) := (0.0, 0.0);
   Vec3      : Vector (1 .. 3) := (0.0, 0.0, 0.0);
   Bool1     : Boolean_Vector (1 .. 1) := (others => False);
   Bool2     : Boolean_Vector (1 .. 2) := (False, False);
   Bool3     : Boolean_Vector (1 .. 3) := (False, False, False);
   Converged : Boolean;
   Success   : Boolean;
   Caught    : Boolean;
begin
   Put_Line ("========================================");
   Put_Line ("Almeida-Pineda Algorithm Test Suite");
   Put_Line ("========================================");

   -- TEST 1 — Sigmoid Math Edge Cases
   Put_Line ("TEST 1 — Sigmoid Math Edge Cases");
   Check ("1.1 Sigmoid(0) equals 0.5", Is_Close (Sigmoid (0.0), 0.5));
   Check ("1.2 Sigmoid(100) clamps to 1.0", Is_Close (Sigmoid (100.0), 1.0));
   Check ("1.3 Sigmoid(-100) clamps to 0.0", Is_Close (Sigmoid (-100.0), 0.0));

   -- TEST 2 — Sigmoid Derivative Math
   Put_Line ("TEST 2 — Sigmoid Derivative Math");
   Check ("2.1 Deriv(0.5) equals 0.25", Is_Close (Sigmoid_Derivative (0.5), 0.25));
   Check ("2.2 Deriv(1.0) equals 0.0", Is_Close (Sigmoid_Derivative (1.0), 0.0));
   Check ("2.3 Deriv(0.0) equals 0.0", Is_Close (Sigmoid_Derivative (0.0), 0.0));

   -- TEST 3 — Dimension Error Handling on Forward Pass
   Put_Line ("TEST 3 — Dimension Error Handling (Forward)");
   Caught := False;
   begin
      Forward_Relaxation (Net2, Vec3, 0.001, 10, Synchronous, Converged);
   exception
      when Dimension_Error => Caught := True;
      when others => null;
   end;
   Check ("3.1 Reject too long input vector", Caught);
   
   Caught := False;
   begin
      Forward_Relaxation (Net2, Vec1, 0.001, 10, Synchronous, Converged);
   exception
      when Dimension_Error => Caught := True;
      when others => null;
   end;
   Check ("3.2 Reject too short input vector", Caught);
   Check ("3.3 Network unchanged on failure", Net2.States (1) = 0.0);

   -- TEST 4 — Dimension Error Handling on Backward Pass
   Put_Line ("TEST 4 — Dimension Error Handling (Backward)");
   Caught := False;
   begin
      Backward_Relaxation (Net2, Vec3, Bool2, 0.001, 10, Synchronous, Converged);
   exception
      when Dimension_Error => Caught := True;
      when others => null;
   end;
   Check ("4.1 Reject target vector length mismatch", Caught);
   
   Caught := False;
   begin
      Backward_Relaxation (Net2, Vec2, Bool3, 0.001, 10, Synchronous, Converged);
   exception
      when Dimension_Error => Caught := True;
      when others => null;
   end;
   Check ("4.2 Reject target mask length mismatch", Caught);
   Check ("4.3 Errors unchanged on failure", Net2.Errors (1) = 0.0);

   -- TEST 5 — Forward Relaxation (Zero weights, Synchronous)
   Put_Line ("TEST 5 — Forward Relaxation (Zero Weights, Synchronous)");
   Net2.States := (0.0, 0.0);
   Forward_Relaxation (Net2, (0.0, 1.0), 0.001, 10, Synchronous, Converged);
   Check ("5.1 Converges instantly with zero weights", Converged);
   Check ("5.2 State(1) evaluates to Sigmoid(0) = 0.5", Is_Close (Net2.States (1), 0.5));
   Check ("5.3 State(2) evaluates to Sigmoid(1) > 0.73", Net2.States (2) > 0.73);

   -- TEST 6 — Forward Relaxation (Zero weights, Asynchronous)
   Put_Line ("TEST 6 — Forward Relaxation (Zero Weights, Asynchronous)");
   Net2.States := (0.0, 0.0);
   Forward_Relaxation (Net2, (0.0, 1.0), 0.001, 10, Asynchronous, Converged);
   Check ("6.1 Converges instantly with zero weights", Converged);
   Check ("6.2 State(1) evaluates to Sigmoid(0) = 0.5", Is_Close (Net2.States (1), 0.5));
   Check ("6.3 State(2) evaluates to Sigmoid(1) > 0.73", Net2.States (2) > 0.73);

   -- TEST 7 — Forward Relaxation (Simple Propagation)
   Put_Line ("TEST 7 — Forward Relaxation (Propagation)");
   Net2.States := (0.0, 0.0);
   Net2.Weights (2, 1) := 10.0; -- Node 1 strongly excites Node 2
   Forward_Relaxation (Net2, (10.0, 0.0), 0.001, 20, Synchronous, Converged);
   Check ("7.1 Converges with simple propagation", Converged);
   Check ("7.2 Node 1 heavily excited (> 0.99)", Net2.States (1) > 0.99);
   Check ("7.3 Node 2 heavily excited due to Node 1 (> 0.99)", Net2.States (2) > 0.99);

   -- TEST 8 — Backward Relaxation (Zero weights)
   Put_Line ("TEST 8 — Backward Relaxation (Zero Weights)");
   Net2.Weights := (others => (others => 0.0));
   Net2.States := (0.5, 0.5); 
   -- Target for 2 is 1.0, Injection = 0.5, Deriv = 0.25, Error = 0.125
   Backward_Relaxation (Net2, (0.0, 1.0), (False, True), 0.001, 10, Synchronous, Converged);
   Check ("8.1 Converges instantly", Converged);
   Check ("8.2 Error(1) is 0 (no target)", Is_Close (Net2.Errors (1), 0.0));
   Check ("8.3 Error(2) computes correctly to 0.125", Is_Close (Net2.Errors (2), 0.125));

   -- TEST 9 — Backward Relaxation (Propagation)
   Put_Line ("TEST 9 — Backward Relaxation (Propagation)");
   Net2.Weights := (others => (others => 0.0));
   Net2.Weights (2, 1) := 1.0; -- Forward weight 1->2. Backward error 2->1.
   Net2.States := (0.5, 0.5); 
   Backward_Relaxation (Net2, (0.0, 1.0), (False, True), 0.001, 10, Synchronous, Converged);
   Check ("9.1 Converges with error propagation", Converged);
   Check ("9.2 Error(2) is non-zero (direct injection)", Net2.Errors (2) > 0.0);
   Check ("9.3 Error(1) receives propagated error from 2", Net2.Errors (1) > 0.0);

   -- TEST 10 — Weight Updates (Gradient direction)
   Put_Line ("TEST 10 — Weight Updates");
   Net2.Errors := (0.5, 0.0);
   Net2.States := (0.0, 0.5);
   Net2.Weights := (others => (others => 0.0));
   Net2.Biases := (others => 0.0);
   Update_Weights (Net2, 0.1);
   -- delta W(1,2) = 0.1 * Error(1) * State(2) = 0.1 * 0.5 * 0.5 = 0.025
   Check ("10.1 W(1,2) increases correctly", Is_Close (Net2.Weights (1, 2), 0.025));
   Check ("10.2 Bias(1) increases by LR * Error(1)", Is_Close (Net2.Biases (1), 0.05));
   Check ("10.3 W(2,1) untouched (Error(2) = 0)", Is_Close (Net2.Weights (2, 1), 0.0));

   -- TEST 11 — End-to-End Train Pattern (Success)
   Put_Line ("TEST 11 — End-to-End Train Pattern (Success)");
   Net2.Weights := (others => (others => 0.0));
   Train_Pattern (
      Net2, (1.0, 0.0), (0.0, 1.0), (False, True), 
      0.1, 0.001, 100, Asynchronous, Success
   );
   Check ("11.1 Train_Pattern reports success", Success);
   Check ("11.2 Weights were modified", Net2.Weights (2, 1) /= 0.0 or Net2.Weights (1, 1) /= 0.0);
   Check ("11.3 Biases were modified", Net2.Biases (2) /= 0.0);

   -- TEST 12 — End-to-End Train Pattern (Failure / Divergence)
   Put_Line ("TEST 12 — End-to-End Train Pattern (Non-convergence)");
   Net2.Weights := (others => (others => 0.0));
   Net2.Biases := (others => 0.0);
   Train_Pattern (
      Net2, (100.0, 100.0), (0.0, 1.0), (True, True), 
      0.1, 0.0, 1, Synchronous, Success -- Max_Epochs = 1, Tolerance = 0 guarantees fail
   );
   Check ("12.1 Train_Pattern gracefully reports failure", not Success);
   Check ("12.2 Weights are unmodified on failure", Is_Close (Net2.Weights (1, 2), 0.0));
   Check ("12.3 Biases are unmodified on failure", Is_Close (Net2.Biases (1), 0.0));

   -- TEST 13 — Single Node Network
   Put_Line ("TEST 13 — Single Node Network (Edge Case)");
   Net1.Weights (1, 1) := 0.5;
   Train_Pattern (
      Net1, Vec1, (1 => 1.0), (1 => True), 
      0.1, 0.001, 100, Synchronous, Success
   );
   Check ("13.1 Single node network trains successfully", Success);
   Check ("13.2 Error was computed and applied", Net1.Errors (1) /= 0.0);
   Check ("13.3 Self-recurrent weight updated", Net1.Weights (1, 1) /= 0.5);

   Put_Line ("");
   Put_Line ("=== " & Natural'Image (Pass_Count) & " passed, "
             & Natural'Image (Fail_Count) & " failed ===");
   pragma Assert (Fail_Count = 0, "Some tests failed");
end Tests;
