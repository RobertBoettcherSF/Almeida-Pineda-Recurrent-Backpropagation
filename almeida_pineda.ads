package Almeida_Pineda is
   pragma Preelaborate;

   -- Strong typing for algorithm-specific data
   type Real is digits 15;
   type Node_Count is new Positive;
   subtype Node_Index is Node_Count;

   -- Vectors and matrices indexed by Node_Index
   type Vector is array (Node_Index range <>) of Real;
   type Matrix is array (Node_Index range <>, Node_Index range <>) of Real;
   type Boolean_Vector is array (Node_Index range <>) of Boolean;

   -- Network state containing weights, biases, states, and error signals
   type Network (Size : Node_Count) is record
      Weights : Matrix (1 .. Size, 1 .. Size) := (others => (others => 0.0));
      Biases  : Vector (1 .. Size) := (others => 0.0);
      States  : Vector (1 .. Size) := (others => 0.0);
      Errors  : Vector (1 .. Size) := (others => 0.0);
   end record;

   -- Update modes for the recurrent relaxation process
   -- Synchronous: All states update simultaneously using t-1 values
   -- Asynchronous: States update sequentially in-place, often converging faster (Gauss-Seidel)
   type Update_Mode is (Synchronous, Asynchronous);

   -- Exception raised when input/target vectors do not match network dimensions
   Dimension_Error : exception;

   -- Standard activation function and its derivative
   function Sigmoid (X : Real) return Real
     with Global => null;

   function Sigmoid_Derivative (Output : Real) return Real
     with Global => null,
          Pre    => Output >= 0.0 and Output <= 1.0;

   -- Forward pass: Run network dynamics until states stabilize (stable fixed point)
   procedure Forward_Relaxation
     (Net        : in out Network;
      Inputs     : in     Vector;
      Tolerance  : in     Real;
      Max_Epochs : in     Positive;
      Mode       : in     Update_Mode;
      Converged  :    out Boolean)
     with Global => null,
          Pre    => Tolerance >= 0.0;

   -- Backward pass: Run reciprocal error network dynamics until error signals stabilize
   procedure Backward_Relaxation
     (Net         : in out Network;
      Targets     : in     Vector;
      Has_Target  : in     Boolean_Vector;
      Tolerance   : in     Real;
      Max_Epochs  : in     Positive;
      Mode        : in     Update_Mode;
      Converged   :    out Boolean)
     with Global => null,
          Pre    => Tolerance >= 0.0;

   -- Apply weight and bias updates based on converged states and errors
   procedure Update_Weights
     (Net           : in out Network;
      Learning_Rate : in     Real)
     with Global => null,
          Pre    => Learning_Rate >= 0.0;

   -- Train a single pattern by running forward, backward, and applying updates
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
     with Global => null,
          Pre    => Learning_Rate >= 0.0 and Tolerance >= 0.0;

end Almeida_Pineda;
