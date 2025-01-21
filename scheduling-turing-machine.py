from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Direction(Enum):
    """Direction for tape head movement"""

    LEFT = "L"
    RIGHT = "R"


@dataclass
class Transition:
    """Represents a single transition in the Turing machine"""

    next_state: str
    write_symbol: str
    direction: Direction


class TapeCell:
    """Represents a single cell on the Turing machine tape"""

    def __init__(self, value: str):
        self.value = value

    def is_blank(self) -> bool:
        return self.value == "B"

    def __str__(self) -> str:
        return str(self.value)


class TuringMachineState:
    """Represents a state in the Turing machine"""

    def __init__(self, name: str, is_final: bool = False):
        self.name = name
        self.is_final = is_final
        self.transitions: Dict[str, Transition] = {}

    def add_transition(
        self, read_symbol: str, next_state: str, write_symbol: str, direction: Direction
    ):
        """Add a transition rule for this state"""
        self.transitions[read_symbol] = Transition(next_state, write_symbol, direction)

    def get_transition(self, symbol: str) -> Optional[Transition]:
        """Get transition for a given symbol"""
        return self.transitions.get(symbol)


class SchedulingTuringMachine:
    """Implementation of a Turing machine for scheduling problems"""

    def __init__(self, num_machines: int = 5, production_rate: int = 10):
        self.num_machines = num_machines
        self.production_rate = production_rate

        # Initialize machine components
        self.tape: List[TapeCell] = []
        self.head_position = 0
        self.states: Dict[str, TuringMachineState] = {}
        self.current_state: TuringMachineState = None

        # Initialize scheduling data
        self.machine_times = [0] * num_machines
        self.machine_schedules = [[] for _ in range(num_machines)]

        # Set up the Turing machine states and transitions
        self._setup_states()

        # Buffer for building numbers
        self.current_number: List[str] = []

    def _setup_states(self):
        """Set up all states and transitions for the Turing machine"""
        # Create states
        state_configs = {
            "START": False,  # Initial state
            "READ": False,  # Reading digits
            "MARK": False,  # Marking processed
            "NEXT": False,  # Moving to next number
            "FINAL": True,  # Final state
        }

        # Initialize all states
        for state_name, is_final in state_configs.items():
            self.states[state_name] = TuringMachineState(state_name, is_final)

        # Define all possible symbols
        digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # Add transitions for START state
        self.states["START"].add_transition("B", "FINAL", "B", Direction.RIGHT)
        self.states["START"].add_transition("#", "READ", "#", Direction.RIGHT)
        for digit in digits:
            self.states["START"].add_transition(digit, "READ", digit, Direction.RIGHT)

        # Add transitions for READ state
        for digit in digits:
            self.states["READ"].add_transition(digit, "READ", digit, Direction.RIGHT)
        self.states["READ"].add_transition("#", "MARK", "#", Direction.LEFT)
        self.states["READ"].add_transition("B", "MARK", "B", Direction.LEFT)

        # Add transitions for MARK state
        for digit in digits:
            self.states["MARK"].add_transition(digit, "MARK", "*", Direction.LEFT)
        self.states["MARK"].add_transition("#", "NEXT", "#", Direction.RIGHT)

        # Add transitions for NEXT state
        for digit in digits:
            self.states["NEXT"].add_transition(digit, "READ", digit, Direction.RIGHT)
        self.states["NEXT"].add_transition("#", "READ", "#", Direction.RIGHT)
        self.states["NEXT"].add_transition("B", "FINAL", "B", Direction.RIGHT)
        self.states["NEXT"].add_transition("*", "NEXT", "*", Direction.RIGHT)

        # Set initial state
        self.current_state = self.states["START"]

    def _find_best_machine(self, order_size: int) -> int:
        """Find the machine with earliest availability"""
        min_time = float("inf")
        best_machine = 0

        for i in range(self.num_machines):
            if self.machine_times[i] < min_time:
                min_time = self.machine_times[i]
                best_machine = i

        # Calculate processing time and update machine schedule
        processing_time = order_size / self.production_rate
        self.machine_times[best_machine] += processing_time
        self.machine_schedules[best_machine].append((order_size, min_time))

        logger.debug(f"Assigned order {order_size} to machine {best_machine}")
        return best_machine

    def _prepare_tape(self, orders: List[int]) -> List[TapeCell]:
        """Prepare the tape with input orders"""
        tape = []

        for order in orders:
            # Add separator before each number
            tape.append(TapeCell("#"))
            # Convert number to digits and add to tape
            for digit in str(order):
                tape.append(TapeCell(digit))

        # Add blank symbol at the end
        tape.append(TapeCell("B"))

        return tape

    def process_orders(self, orders: List[int]) -> Dict:
        """Process a list of orders through the Turing machine"""
        # Initialize/reset machine state
        self.tape = self._prepare_tape(orders)
        self.head_position = 0
        self.current_state = self.states["START"]
        self.current_number = []

        logger.info("Starting order processing...")

        # Main processing loop
        while not self.current_state.is_final:
            current_symbol = self.tape[self.head_position].value

            # Get next transition
            transition = self.current_state.get_transition(current_symbol)
            if not transition:
                raise Exception(
                    f"No valid transition for state '{self.current_state.name}' "
                    f"and symbol '{current_symbol}'"
                )

            # Process current symbol based on state
            if self.current_state.name == "READ":
                if current_symbol.isdigit():
                    self.current_number.append(current_symbol)

            elif self.current_state.name == "MARK" and self.current_number:
                # Process completed number
                order_size = int("".join(self.current_number))
                self._find_best_machine(order_size)
                self.current_number = []

            # Apply transition
            self.tape[self.head_position].value = transition.write_symbol
            self.current_state = self.states[transition.next_state]
            self.head_position += 1 if transition.direction == Direction.RIGHT else -1

            logger.debug(
                f"State: {self.current_state.name}, "
                f"Symbol: {current_symbol}, "
                f"Position: {self.head_position}"
            )

        logger.info("Order processing complete.")
        return self._generate_schedule()

    def _generate_schedule(self) -> Dict:
        """Generate the final schedule report"""
        schedule = {}
        for i, machine_schedule in enumerate(self.machine_schedules):
            schedule[f"Machine_{i+1}"] = {
                "orders": [
                    {"size": size, "start_time": start}
                    for size, start in machine_schedule
                ],
                "total_time": self.machine_times[i],
            }
        return schedule


def print_schedule(schedule: Dict):
    """Print the schedule in a formatted way"""
    print("\nProduction Schedule:")
    print("=" * 50)

    for machine_id, details in schedule.items():
        print(f"\n{machine_id}:")
        print(f"Total time: {details['total_time']:.2f} minutes")
        if details["orders"]:
            print("Orders:")
            for order in details["orders"]:
                print(f"  Size: {order['size']:4d} | Start: {order['start_time']:6.2f}")
        else:
            print("No orders assigned")


def main():
    # Test orders
    orders = [50, 100, 200, 20, 10, 30, 1000, 500, 800, 80, 70]

    try:
        # Create and run Turing machine
        tm = SchedulingTuringMachine()
        schedule = tm.process_orders(orders)
        print(f"Orders: {orders}")
        print_schedule(schedule)
        print()
    except Exception as e:
        logger.error(f"Error processing orders: {str(e)}")


if __name__ == "__main__":
    main()
