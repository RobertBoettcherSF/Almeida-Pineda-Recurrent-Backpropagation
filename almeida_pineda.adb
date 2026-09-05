with Ada.Numerics.Generic_Elementary_Functions;

package body Almeida_Pineda is
   package Math is new Ada.Numerics.Generic_Elementary_Functions (Real);
   use Math;

   -----------------------------------------------------------------------------
   -- Activation Functions
   -----------------------------------------------------------------------------
   function Sigmoid (X : Real) return Real is
   begin
      -- Protect against floating-point overflow/underflow in Exp
      if X > 50.0 then
         return 1.0;
      elsif X < -50.0 then
         return 0.0;
      end if;
      return 1.0 / (1.0 + Exp (-X));
   end Sigmoid;

   function Sigmoid_Derivative (Output : Real) return Real is
   begin
      -- Standard derivative for Sigmoid: f'(x) = f(x) * (1 - f(x))
      -- Operates on the already-calculated Output, not the pre-activation sum
      return Output * (1.0 - Output);
   end Sigmoid_Derivative;

   -----------------------------------------------------------------------------
   -- Forward Relaxation (Finding the steady state)
   -----------------------------------------------------------------------------
   procedure Forward_Relaxation
     (Net        : in out Network;
      Inputs     : in     Vector;
      Tolerance  : in     Real;
      Max_Epochs : in     Positive;
      Mode       : in     Update_Mode;
      Converged  :    out Boolean)
   is
      Next_States : Vector (1 .. Net.Size) := Net.States;
      Max_Diff    : Real;
      Sum         : Real;
      Old_State   : Real;
   begin
      -- Validate input constraints dynamically
      if Inputs'Length /= Net.Size or else Inputs'First /= 1 then
         raise Dimension_Error;
      end if;

      Converged := False;

      -- Iterate until the network states stabilize (fixed point)
      for Epoch in 1 .. Max_Epochs loop
         Max_Diff := 0.0;

         for I in 1 .. Net.Size loop
            Sum := Net.Biases (I) + Inputs (I);
            
            for J in 1 .. Net.Size loop
               if Mode = Synchronous then
                  -- Use entirely previous-epoch states
                  Sum := Sum + Net.Weights (I, J) * Net.States (J);
               else
                  -- Asynchronous (Gauss-Seidel): Use newest available states
                  Sum := Sum + Net.Weights (I, J) * Next_States (J);
               end if;
            end loop;

            Old_State := (if Mode = Synchronous then Net.States (I) else Next_States (I));
            Next_States (I) := Sigmoid (Sum);

            if abs (Next_States (I) - Old_State) > Max_Diff then
               Max_Diff := abs (Next_States (I) - Old_State);
            end if;
         end loop;

         Net.States := Next_States;

         if Max_Diff <= Tolerance then
            Converged := True;
            exit;
         end if;
      end loop;
   end Forward_Relaxation;

   -----------------------------------------------------------------------------
   -- Backward Relaxation (Finding the steady state of the error network)
   -----------------------------------------------------------------------------
   procedure Backward_Relaxation
     (Net         : in out Network;
      Targets     : in     Vector;
      Has_Target  : in     Boolean_Vector;
      Tolerance   : in     Real;
      Max_Epochs  : in     Positive;
      Mode        : in     Update_Mode;
      Converged   :    out Boolean)
   is
      Next_Errors : Vector (1 .. Net.Size) := Net.Errors;
      Max_Diff    : Real;
      Sum         : Real;
      Old_Error   : Real;
      Injection   : Real;
      Deriv       : Real;
   begin
      -- Validate bounds dynamically
      if Targets'Length /= Net.Size or else Targets'First /= 1 or else
         Has_Target'Length /= Net.Size or else Has_Target'First /= 1
      then
         raise Dimension_Error;
      end if;

      Converged := False;

      -- Iterate the transposed linear network until errors stabilize
      for Epoch in 1 .. Max_Epochs loop
         Max_Diff := 0.0;

         for I in 1 .. Net.Size loop
            Sum := 0.0;
            for J in 1 .. Net.Size loop
               -- Crucial Almeida-Pineda step: Use transposed weights W(J, I)
               if Mode = Synchronous then
                  Sum := Sum + Net.Weights (J, I) * Net.Errors (J);
               else
                  Sum := Sum + Net.Weights (J, I) * Next_Errors (J);
               end if;
            end loop;

            Deriv := Sigmoid_Derivative (Net.States (I));
            
            if Has_Target (I) then
               Injection := Targets (I) - Net.States (I);
            else
               Injection := 0.0;
            end if;

            Old_Error := (if Mode = Synchronous then Net.Errors (I) else Next_Errors (I));
            
            -- Recurrent backprop equation: E_i = f'(net_i) * (sum(w_ji * E_j) + J_i)
            Next_Errors (I) := Deriv * (Sum + Injection);

            if abs (Next_Errors (I) - Old_Error) > Max_Diff then
               Max_Diff := abs (Next_Errors (I) - Old_Error);
            end if;
         end loop;

         Net.Errors := Next_Errors;

         if Max_Diff <= Tolerance then
            Converged := True;
            exit;
         end if;
      end loop;
   end Backward_Relaxation;

   -----------------------------------------------------------------------------
   -- Update Weights
   -----------------------------------------------------------------------------
   procedure Update_Weights
     (Net           : in out Network;
      Learning_Rate : in     Real)
   is
   begin
      for I in 1 .. Net.Size loop
         for J in 1 .. Net.Size loop
            -- Update rule: Delta W_ij = LR * Error_i * State_j
            Net.Weights (I, J) := Net.Weights (I, J) +
                                  Learning_Rate * Net.Errors (I) * Net.States (J);
         end loop;
         Net.Biases (I) := Net.Biases (I) + Learning_Rate * Net.Errors (I);
      end loop;
   end Update_Weights;

   -----------------------------------------------------------------------------
   -- Train Pattern
   -----------------------------------------------------------------------------
   procedure Train_Pattern
     (Net           : in out Network;
      Inputs        : in     Vector;
      Targets       : in     Vector;
      Has_Target    : in     Boolean_Vector;
      Learning_Rate : in     Real;
      Tolerance     : in     Real;
      Max_Epochs    : in     Positive;
      Mode          : in     Update_Mode;
      Success       :    out Boolean)
   is
      Fwd_Conv, Bwd_Conv : Boolean;
   begin
      Forward_Relaxation (Net, Inputs, Tolerance, Max_Epochs, Mode, Fwd_Conv);
      if not Fwd_Conv then
         Success := False;
         return;
      end if;

      Backward_Relaxation (Net, Targets, Has_Target, Tolerance, Max_Epochs, Mode, Bwd_Conv);
      if not Bwd_Conv then
         Success := False;
         return;
      end if;

      Update_Weights (Net, Learning_Rate);
      Success := True;
   end Train_Pattern;

end Almeida_Pineda;
