# Copyright 2021 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import upf
from upf.shortcuts import *

from collections import namedtuple


Example = namedtuple('Example', ['problem', 'plan'])

def get_example_problems():
    problems = {}

    # basic
    x = upf.Fluent('x')
    a = upf.Action('a')
    a.add_precondition(Not(x))
    a.add_effect(x, True)
    problem = upf.Problem('basic')
    problem.add_fluent(x)
    problem.add_action(a)
    problem.set_initial_value(x, False)
    problem.add_goal(x)
    plan = upf.SequentialPlan([upf.ActionInstance(a)])
    basic = Example(problem=problem, plan=plan)
    problems['basic'] = basic

    # basic conditional
    x = upf.Fluent('x')
    y = upf.Fluent('y')
    a_x = upf.Action('a_x')
    a_y = upf.Action('a_y')
    a_x.add_precondition(Not(x))
    a_x.add_effect(x, True, y)
    a_y.add_precondition(Not(y))
    a_y.add_effect(y, True)
    problem = upf.Problem('basic_conditional')
    problem.add_fluent(x)
    problem.add_fluent(y)
    problem.add_action(a_x)
    problem.add_action(a_y)
    problem.set_initial_value(x, False)
    problem.set_initial_value(y, False)
    problem.add_goal(x)
    plan = upf.SequentialPlan([upf.ActionInstance(a_y), upf.ActionInstance(a_x)])
    basic_conditional = Example(problem=problem, plan=plan)
    problems['basic_conditional'] = basic_conditional

    # complex conditional
    a = upf.Fluent('a')
    b = upf.Fluent('b')
    c = upf.Fluent('c')
    d = upf.Fluent('d')
    k = upf.Fluent('k')
    x = upf.Fluent('x')
    y = upf.Fluent('y')
    z = upf.Fluent('z')
    a_act = upf.Action('A')
    a_0_act = upf.Action('A_0')
    a_1_act = upf.Action('A_1')
    a_2_act = upf.Action('A_2')
    a_act.add_precondition(Not(a))
    a_act.add_effect(a, TRUE())
    a_act.add_effect(k, TRUE(), b)
    a_act.add_effect(x, TRUE(), Not(c))
    a_act.add_effect(y, FALSE(), d)
    a_0_act.add_precondition(Not(a))
    a_0_act.add_precondition(d)
    a_0_act.add_effect(b, TRUE())
    a_1_act.add_precondition(Not(a))
    a_1_act.add_precondition(d)
    a_1_act.add_precondition(b)
    a_1_act.add_effect(c, FALSE(), c)
    a_1_act.add_effect(c, TRUE(), Not(c))
    a_2_act.add_effect(a, FALSE())
    a_2_act.add_effect(d, TRUE())
    a_2_act.add_effect(z, FALSE(), z)
    a_2_act.add_effect(z, TRUE(), Not(z))
    problem = upf.Problem('complex_conditional')
    problem.add_fluent(a)
    problem.add_fluent(b)
    problem.add_fluent(c)
    problem.add_fluent(d)
    problem.add_fluent(k)
    problem.add_fluent(x)
    problem.add_fluent(y)
    problem.add_fluent(z)
    problem.add_action(a_act)
    problem.add_action(a_0_act)
    problem.add_action(a_1_act)
    problem.add_action(a_2_act)
    problem.set_initial_value(a, True)
    problem.set_initial_value(b, False)
    problem.set_initial_value(c, True)
    problem.set_initial_value(d, False)
    problem.set_initial_value(k, False)
    problem.set_initial_value(x, False)
    problem.set_initial_value(y, True)
    problem.set_initial_value(z, False)
    problem.add_goal(a)
    problem.add_goal(b)
    problem.add_goal(Not(c))
    problem.add_goal(d)
    problem.add_goal(k)
    problem.add_goal(x)
    problem.add_goal(Not(y))
    problem.add_goal(z)
    plan = upf.SequentialPlan([upf.ActionInstance(a_2_act),
                                upf.ActionInstance(a_0_act),
                                upf.ActionInstance(a_1_act),
                                upf.ActionInstance(a_act)])
    complex_conditional = Example(problem=problem, plan=plan)
    problems['complex_conditional'] = complex_conditional

    # basic exists
    sem = UserType("Semaphore")
    x = upf.Fluent('x')
    y = upf.Fluent('y', BoolType(), [sem])
    o1 = upf.Object('o1', sem)
    o2 = upf.Object('o2', sem)
    s_var = upf.Variable("s", sem)
    a = upf.Action('a')
    a.add_precondition(Exists(FluentExp(y, [s_var]), s_var))
    a.add_effect(x, True)
    problem = upf.Problem('basic_exists')
    problem.add_fluent(x)
    problem.add_fluent(y)
    problem.add_object(o1)
    problem.add_object(o2)
    problem.add_action(a)
    problem.set_initial_value(x, False)
    problem.set_initial_value(y(o1), True)
    problem.set_initial_value(y(o2), False)
    problem.add_goal(x)
    plan = upf.SequentialPlan([upf.ActionInstance(a)])
    basic_exists = Example(problem=problem, plan=plan)
    problems['basic_exists'] = basic_exists


    # basic forall
    sem = UserType("Semaphore")
    x = upf.Fluent('x')
    y = upf.Fluent('y', BoolType(), [sem])
    o1 = upf.Object('o1', sem)
    o2 = upf.Object('o2', sem)
    s_var = upf.Variable("s", sem)
    a = upf.Action('a')
    a.add_precondition(Forall(Not(FluentExp(y, [s_var])), s_var))
    a.add_effect(x, True)
    problem = upf.Problem('basic_exists')
    problem.add_fluent(x)
    problem.add_fluent(y)
    problem.add_object(o1)
    problem.add_object(o2)
    problem.add_action(a)
    problem.set_initial_value(x, False)
    problem.set_initial_value(y(o1), False)
    problem.set_initial_value(y(o2), False)
    problem.add_goal(x)
    plan = upf.SequentialPlan([upf.ActionInstance(a)])
    basic_forall = Example(problem=problem, plan=plan)
    problems['basic_forall'] = basic_forall

    # robot
    Location = UserType('Location')
    robot_at = upf.Fluent('robot_at', BoolType(), [Location])
    battery_charge = upf.Fluent('battery_charge', RealType(0, 100))
    move = upf.Action('move', l_from=Location, l_to=Location)
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(GE(battery_charge, 10))
    move.add_precondition(Not(Equals(l_from, l_to)))
    move.add_precondition(robot_at(l_from))
    move.add_precondition(Not(robot_at(l_to)))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)
    move.add_effect(battery_charge, Minus(battery_charge, 10))
    l1 = upf.Object('l1', Location)
    l2 = upf.Object('l2', Location)
    problem = upf.Problem('robot')
    problem.add_fluent(robot_at)
    problem.add_fluent(battery_charge)
    problem.add_action(move)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.set_initial_value(robot_at(l1), True)
    problem.set_initial_value(robot_at(l2), False)
    problem.set_initial_value(battery_charge, 100)
    problem.add_goal(robot_at(l2))
    plan = upf.SequentialPlan([upf.ActionInstance(move, [ObjectExp(l1), ObjectExp(l2)])])
    robot = Example(problem=problem, plan=plan)
    problems['robot'] = robot

    # robot decrease
    Location = UserType('Location')
    robot_at = upf.Fluent('robot_at', BoolType(), [Location])
    battery_charge = upf.Fluent('battery_charge', RealType(0, 100))
    move = upf.Action('move', l_from=Location, l_to=Location)
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(GE(battery_charge, 10))
    move.add_precondition(Not(Equals(l_from, l_to)))
    move.add_precondition(robot_at(l_from))
    move.add_precondition(Not(robot_at(l_to)))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)
    move.add_decrease_effect(battery_charge, 10)
    l1 = upf.Object('l1', Location)
    l2 = upf.Object('l2', Location)
    problem = upf.Problem('robot_decrease')
    problem.add_fluent(robot_at)
    problem.add_fluent(battery_charge)
    problem.add_action(move)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.set_initial_value(robot_at(l1), True)
    problem.set_initial_value(robot_at(l2), False)
    problem.set_initial_value(battery_charge, 100)
    problem.add_goal(robot_at(l2))
    plan = upf.SequentialPlan([upf.ActionInstance(move, [ObjectExp(l1), ObjectExp(l2)])])
    robot_decrease = Example(problem=problem, plan=plan)
    problems['robot_decrease'] = robot_decrease

    # robot_loader
    Location = UserType('Location')
    robot_at = upf.Fluent('robot_at', BoolType(), [Location])
    cargo_at = upf.Fluent('cargo_at', BoolType(), [Location])
    cargo_mounted = upf.Fluent('cargo_mounted')
    move = upf.Action('move', l_from=Location, l_to=Location)
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(Not(Equals(l_from, l_to)))
    move.add_precondition(robot_at(l_from))
    move.add_precondition(Not(robot_at(l_to)))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)
    load = upf.Action('load',loc=Location)
    loc = load.parameter('loc')
    load.add_precondition(cargo_at(loc))
    load.add_precondition(robot_at(loc))
    load.add_precondition(Not(cargo_mounted))
    load.add_effect(cargo_at(loc), False)
    load.add_effect(cargo_mounted, True)
    unload = upf.Action('unload', loc=Location)
    loc = unload.parameter('loc')
    unload.add_precondition(Not(cargo_at(loc)))
    unload.add_precondition(robot_at(loc))
    unload.add_precondition(cargo_mounted)
    unload.add_effect(cargo_at(loc), True)
    unload.add_effect(cargo_mounted, False)
    l1 = upf.Object('l1', Location)
    l2 = upf.Object('l2', Location)
    problem = upf.Problem('robot_loader')
    problem.add_fluent(robot_at)
    problem.add_fluent(cargo_at)
    problem.add_fluent(cargo_mounted)
    problem.add_action(move)
    problem.add_action(load)
    problem.add_action(unload)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.set_initial_value(robot_at(l1), True)
    problem.set_initial_value(robot_at(l2), False)
    problem.set_initial_value(cargo_at(l1), False)
    problem.set_initial_value(cargo_at(l2), True)
    problem.set_initial_value(cargo_mounted, False)
    problem.add_goal(cargo_at(l1))
    plan = upf.SequentialPlan([upf.ActionInstance(move, [ObjectExp(l1), ObjectExp(l2)]),
                               upf.ActionInstance(load, [ObjectExp(l2)]),
                               upf.ActionInstance(move, [ObjectExp(l2), ObjectExp(l1)]),
                               upf.ActionInstance(unload, [ObjectExp(l1)])])
    robot_loader = Example(problem=problem, plan=plan)
    problems['robot_loader'] = robot_loader

    # robot_loader_adv
    Robot = UserType('Robot')
    Container = UserType('Container')
    Location = UserType('Location')
    robot_at = upf.Fluent('robot_at', BoolType(), [Robot, Location])
    cargo_at = upf.Fluent('cargo_at', BoolType(), [Container, Location])
    cargo_mounted = upf.Fluent('cargo_mounted', BoolType(), [Container, Robot])
    move = upf.Action('move', l_from=Location, l_to=Location, r=Robot)
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    r = move.parameter('r')
    move.add_precondition(Not(Equals(l_from, l_to)))
    move.add_precondition(robot_at(r, l_from))
    move.add_precondition(Not(robot_at(r, l_to)))
    move.add_effect(robot_at(r, l_from), False)
    move.add_effect(robot_at(r, l_to), True)
    load = upf.Action('load', loc=Location, r=Robot, c=Container)
    loc = load.parameter('loc')
    r = load.parameter('r')
    c = load.parameter('c')
    load.add_precondition(cargo_at(c, loc))
    load.add_precondition(robot_at(r, loc))
    load.add_precondition(Not(cargo_mounted(c, r)))
    load.add_effect(cargo_at(c, loc), False)
    load.add_effect(cargo_mounted(c,r), True)
    unload = upf.Action('unload', loc=Location, r=Robot, c=Container)
    loc = unload.parameter('loc')
    r = unload.parameter('r')
    c = unload.parameter('c')
    unload.add_precondition(Not(cargo_at(c, loc)))
    unload.add_precondition(robot_at(r, loc))
    unload.add_precondition(cargo_mounted(c,r))
    unload.add_effect(cargo_at(c, loc), True)
    unload.add_effect(cargo_mounted(c,r), False)
    l1 = upf.Object('l1', Location)
    l2 = upf.Object('l2', Location)
    l3 = upf.Object('l3', Location)
    r1 = upf.Object('r1', Robot)
    c1 = upf.Object('c1', Container)
    problem = upf.Problem('robot_loader_adv')
    problem.add_fluent(robot_at)
    problem.add_fluent(cargo_at)
    problem.add_fluent(cargo_mounted)
    problem.add_action(move)
    problem.add_action(load)
    problem.add_action(unload)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_object(l3)
    problem.add_object(r1)
    problem.add_object(c1)
    problem.set_initial_value(robot_at(r1,l1), True)
    problem.set_initial_value(robot_at(r1,l2), False)
    problem.set_initial_value(robot_at(r1,l3), False)
    problem.set_initial_value(cargo_at(c1,l1), False)
    problem.set_initial_value(cargo_at(c1,l2), True)
    problem.set_initial_value(cargo_at(c1,l3), False)
    problem.set_initial_value(cargo_mounted(c1,r1), False)
    problem.add_goal(cargo_at(c1,l3))
    problem.add_goal(robot_at(r1,l1))
    plan = upf.SequentialPlan([upf.ActionInstance(move, [ObjectExp(l1), ObjectExp(l2), ObjectExp(r1)]),
                               upf.ActionInstance(load, [ObjectExp(l2), ObjectExp(r1), ObjectExp(c1)]),
                               upf.ActionInstance(move, [ObjectExp(l2), ObjectExp(l3), ObjectExp(r1)]),
                               upf.ActionInstance(unload, [ObjectExp(l3), ObjectExp(r1), ObjectExp(c1)]),
                               upf.ActionInstance(move, [ObjectExp(l3), ObjectExp(l1), ObjectExp(r1)])])
    robot_loader_adv = Example(problem=problem, plan=plan)
    problems['robot_loader_adv'] = robot_loader_adv

    #robot connected locations
    Location = UserType('Location')
    Robot = UserType('Robot')
    is_at = upf.Fluent('is_at', BoolType(), [Location, Robot])
    battery_charge = upf.Fluent('battery_charge', RealType(0, 100), [Robot])
    is_connected = upf.Fluent('is_connected', BoolType(), [Location, Location])
    move = upf.Action('move', robot=Robot, l_from=Location, l_to=Location)
    robot = move.parameter('robot')
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(GE(battery_charge(robot), 10))
    move.add_precondition(Not(Equals(l_from, l_to)))
    move.add_precondition(is_at(l_from, robot))
    move.add_precondition(Not(is_at(l_to, robot)))
    move.add_precondition(Or(is_connected(l_from, l_to), is_connected(l_to, l_from)))
    move.add_effect(is_at(l_from, robot), False)
    move.add_effect(is_at(l_to, robot), True)
    move.add_decrease_effect(battery_charge(robot), 10)
    move_2 = upf.Action('move_2', robot=Robot, l_from=Location, l_to=Location)
    robot = move_2.parameter('robot')
    l_from = move_2.parameter('l_from')
    l_to = move_2.parameter('l_to')
    move_2.add_precondition(GE(battery_charge(robot), 15))
    move_2.add_precondition(Not(Equals(l_from, l_to)))
    move_2.add_precondition(is_at(l_from, robot))
    move_2.add_precondition(Not(is_at(l_to, robot)))
    mid_location = upf.Variable('mid_loc', Location)
    #(E (location mid_location)
    # !((mid_location == l_from) || (mid_location == l_to)) && (is_connected(l_from, mid_location) || is_connected(mid_location, l_from)) &&
    # && (is_connected(l_to, mid_location) || is_connected(mid_location, l_to)))
    move_2.add_precondition(Exists(And(Not(Or(Equals(mid_location, l_from), Equals(mid_location, l_to))),
                            Or(is_connected(l_from, mid_location), is_connected(mid_location, l_from)),
                            Or(is_connected(l_to, mid_location), is_connected(mid_location, l_to))), mid_location))
    move_2.add_effect(is_at(l_from, robot), False)
    move_2.add_effect(is_at(l_to, robot), True)
    move_2.add_decrease_effect(battery_charge(robot), 15)
    l1 = upf.Object('l1', Location)
    l2 = upf.Object('l2', Location)
    l3 = upf.Object('l3', Location)
    l4 = upf.Object('l4', Location)
    l5 = upf.Object('l5', Location)
    r1 = upf.Object('r1', Robot)
    problem = upf.Problem('robot_connected_locations')
    problem.add_fluent(is_at)
    problem.add_fluent(battery_charge)
    problem.add_fluent(is_connected)
    problem.add_action(move)
    problem.add_action(move_2)
    problem.add_object(r1)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_object(l3)
    problem.add_object(l4)
    problem.add_object(l5)
    problem.set_initial_value(is_at(l1, r1), True)
    problem.set_initial_value(is_at(l2, r1), False)
    problem.set_initial_value(is_at(l3, r1), False)
    problem.set_initial_value(is_at(l4, r1), False)
    problem.set_initial_value(is_at(l5, r1), False)
    problem.set_initial_value(is_connected(l1, l2), True)
    problem.set_initial_value(is_connected(l2, l3), True)
    problem.set_initial_value(is_connected(l3, l4), True)
    problem.set_initial_value(is_connected(l4, l5), True)
    problem.set_initial_value(battery_charge(r1), 100)
    problem.add_goal(is_at(l5, r1))
    plan = upf.SequentialPlan([upf.ActionInstance(move_2, [ObjectExp(r1), ObjectExp(l1), ObjectExp(l3)]),
                                upf.ActionInstance(move_2, [ObjectExp(r1), ObjectExp(l3), ObjectExp(l5)])])
    robot_locations_connected = Example(problem=problem, plan=plan)
    problems['robot_locations_connected'] = robot_locations_connected

    return problems
