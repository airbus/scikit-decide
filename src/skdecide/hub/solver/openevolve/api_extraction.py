import ast
import inspect
import re
import sys
import textwrap
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from inspect import get_annotations
from types import ModuleType
from typing import Any, Generic, Optional, TypeVar, Union, get_type_hints

import skdecide
import skdecide.builders.domain as domain_builders
import skdecide.hub as hub_module
from skdecide import Domain, Space
from skdecide.builders.domain import SingleAgent

T = TypeVar("T")


def _extract_domain_characteristics(domain_cls: type[Domain]) -> dict[str, type[Any]]:
    """Extract domain characteristics defined by builders mixins.

    e.g.
    agent -> SingleAgent
    observability -> FullyObservable
    ...

    """
    characteristics = {}
    for parent_class in domain_cls.__mro__:
        if parent_class.__module__.startswith(domain_builders.__name__):
            characteristic_name = parent_class.__module__.split(".")[-1].capitalize()
            # keep characteristic only if finer that the one already stored
            if characteristic_name not in characteristics or issubclass(
                parent_class, characteristics[characteristic_name]
            ):
                characteristics[characteristic_name] = parent_class

    return characteristics


def _get_characteristic_docstring_summary(mixin: type[Any]) -> str:
    """Extract and clean docstring first line of a builders mixin."""
    firstline = mixin.__doc__.splitlines()[0]
    return firstline.replace("A domain must inherit this class if ", "")


def _format_domain_capabilities_for_prompt(
    domain_capabilities: dict[str, type[Any]],
    subtitle_level: int = 1,
    add_docstring_summary: bool = True,
) -> str:
    """Format for prompt list of domain mixins."""
    summary = (subtitle_level - 1) * "#" + "# Domain Capabilities (High-Level)\n"
    for name, cls in domain_capabilities.items():
        summary += f"- **{name}**: {cls.__name__}"
        if add_docstring_summary:
            summary += f" -> {_get_characteristic_docstring_summary(cls)}"
        summary += "\n"
    summary += "\n"
    return summary


def _extract_domain_base_types(
    domain_cls: type[Domain],
    skip_predefined: bool = True,
    only_predefined: bool = False,
) -> dict[str, type[Any]]:
    """Extract base types T_event, T_state, ... from the domain class

    Skip types that are defined (and unchanged) by characteristic mixins:
    - T_memory -> skdecide.builders.domain.memory
    - T_agent -> skdecide.builders.domain.agent
    - ...

    # Parameters
    domain_cls
    skip_predefined: if True, skip types that are defined (and unchanged) by characteristic mixins
        (T_memory -> skdecide.builders.domain.memory, T_agent -> skdecide.builders.domain.agent, ...)
    only_predefined: if True, skip non-predefined types (state, obs, action, ...)

    # Returns

    """
    characteristics = _extract_domain_characteristics(domain_cls)
    base_types = {}
    for t_name in dir(domain_cls):
        if t_name.startswith("T_"):
            t_value = getattr(domain_cls, t_name)
            is_predefined = any(
                t_name in dir(mixin) and getattr(mixin, t_name) == t_value
                for mixin in characteristics.values()
            )
            if (is_predefined and not skip_predefined) or (
                not is_predefined and not only_predefined
            ):
                base_types[t_name] = t_value
    return base_types


def _is_module_to_document(
    module_name: str,
    domain_cls: Optional[type[Domain]] = None,
    include_skdecide_api: bool = True,
    include_hub_api: bool = True,
    include_domain_cls_module: bool = True,
    user_modules: Optional[list[ModuleType]] = None,
) -> bool:
    return (
        (
            include_skdecide_api
            and module_name.startswith(skdecide.__name__)
            and not module_name.startswith(hub_module.__name__)
        )
        or (include_hub_api and module_name.startswith(hub_module.__name__))
        or (
            include_domain_cls_module
            and domain_cls is not None
            and module_name == domain_cls.__module__
        )
        or user_modules is not None
        and module_name in [mod.__name__ for mod in user_modules]
    )


def is_class_to_document(
    cls: type[Any],
    domain_cls: Optional[type[Domain]] = None,
    include_hub_api: bool = True,
    include_domain_cls_module: bool = True,
    user_modules: Optional[list[ModuleType]] = None,
) -> bool:
    """Check whether the class is to be included in generated doc.

    Used when in recursive mode.

    Filter the class according to the module it came from:
    - from skdecide core: always taken
    - from skecide hub: if `include_hub_api`
    - from the same module as the main domain class `domain_cls`: if `include_domain_cls_module`
    - from another module: if listed in `user_modules`

    # Parameters
    cls
    domain_cls
    include_hub_api
    include_domain_cls_module
    user_modules

    #Returns

    """
    return hasattr(cls, "__module__") and _is_module_to_document(
        cls.__module__,
        domain_cls=domain_cls,
        include_hub_api=include_hub_api,
        include_domain_cls_module=include_domain_cls_module,
        user_modules=user_modules,
    )


def _format_domain_base_types_for_prompt(
    base_types: dict[str, type[Any]],
    subtitle_level: int = 1,
) -> str:
    summary = (subtitle_level - 1) * "#" + "# Domain base types\n"
    for name, value in base_types.items():
        summary += f"- **{name}**: {value}"
        summary += "\n"
    summary += "\n"

    return summary


@dataclass
class ApiExtractionParams:
    """Parameters for api extraction and formating

    # Attributes
    recursive: if True, extract also the api for classes found in method/attribute annotations,
        according to some filters (see `include_hub_api`, `include_domain_cls_module`, and `user_modules`)
    strip_admonitions: whether stripping admonitions like "!!! tip" or "!!! warning" from dosctrings
    extract_observation_space_cls: whether using a domain instance to extract the observation space class
        Can be set to False if `domain.get_observation_space()` is too costly.
    extract_action_space_cls: whether using a domain instance to extract the action space class
        Can be set to False if `domain.get_observation_space()` is too costly.
    simplify_signature: whether simplifying method signatures,
        i.e. replace D.T_agent, T_memory and T_concurrency according to mixins and flatten Union's
    include_hub_api: if `recursive`, extract api for classes defined in submodules from `skdecide.hub`
    include_domain_cls_module: if `recursive`, extract api for classes defined in the same module as the domain class
    user_modules: if `recursive`, extract api for classes defined in the listed modules
    subtitle_level: level of subtitle to use (if nested)

    """

    recursive: bool = True
    strip_admonitions: bool = False
    extract_observation_space_cls: bool = True
    extract_action_space_cls: bool = True
    simplify_signature: bool = True
    include_hub_api: bool = True
    include_domain_cls_module: bool = True
    user_modules: Optional[list[ModuleType]] = field(default_factory=list)
    subtitle_level: int = 2


def generate_public_api(
    cls: type[Any],
    domain_cls: Optional[type[Any]] = None,
    domain: Optional[Domain] = None,
    params: Optional[ApiExtractionParams] = None,
) -> str:
    """Generate a markdown string describing the public api of the class

    # Parameters
    cls: class to document
    domain_cls: domain class for which these apis are required.
        Default to `cls` if it is a domain class.
    domain: if provided, this domain instance is used to detect action space and observation space class
        (in order to expose their api)
    params: other parameters passed to `generate_public_apis()`, see `ApiExtractionParams` doc.

    # Returns

    """
    if domain_cls is None and issubclass(cls, Domain):
        domain_cls = cls
    if params is None:
        params = ApiExtractionParams()
    return generate_public_apis(
        [cls],
        subtitle_level=params.subtitle_level,
        recursive=params.recursive,
        domain_cls=domain_cls,
        domain=domain,
        strip_admonitions=params.strip_admonitions,
        extract_observation_space_cls=params.extract_observation_space_cls,
        extract_action_space_cls=params.extract_action_space_cls,
        simplify_signature=params.simplify_signature,
        include_hub_api=params.include_hub_api,
        include_domain_cls_module=params.include_domain_cls_module,
        user_modules=params.user_modules,
    )


def generate_public_apis(
    classes: list[type[Any]],
    subtitle_level: int = 1,
    recursive: bool = False,
    domain_cls: Optional[type[Any]] = None,
    domain: Optional[Domain] = None,
    strip_admonitions: bool = False,
    extract_observation_space_cls: bool = True,
    extract_action_space_cls: bool = True,
    simplify_signature: bool = True,
    include_hub_api: bool = True,
    include_domain_cls_module: bool = True,
    user_modules: Optional[list[ModuleType]] = None,
) -> str:
    """Generate a markdown string describing the public api of the class

    # Parameters
    classes: classes to document
    subtitle_level: level of subtitle to use (if nested)
    recursive: if True, also document the classes found in method/attribute annotations, that are filtered by `is_class_to_document`
    domain_cls: domain class for which these apis are required
    domain: if provided, this domain instance is used to detect action space and observation space class (in order to expose their api)
    strip_admonitions: whether stripping admonitions like "!!! tip" or "!!! warning" from dosctrings
    extract_observation_space_cls: whether using the domain instance to extract the observation space class
    extract_action_space_cls: whether using the domain instance to extract the action space class
    simplify_signature: whether simplifying method signatures (replacing D.T_agent, T_memory and T_concurrency according to mixins, and flattening Union's)
    include_hub_api: see `is_class_to_document`
    include_domain_cls_module: see `is_class_to_document`
    user_modules: see `is_class_to_document`

    # Returns

    """
    return "\n\n".join(
        [
            format_api_for_prompt(api=api, subtitle_level=subtitle_level)
            for api in get_public_apis(
                classes,
                recursive=recursive,
                domain_cls=domain_cls,
                domain=domain,
                strip_admonitions=strip_admonitions,
                extract_observation_space_cls=extract_observation_space_cls,
                extract_action_space_cls=extract_action_space_cls,
                simplify_signature=simplify_signature,
                include_hub_api=include_hub_api,
                include_domain_cls_module=include_domain_cls_module,
                user_modules=user_modules,
            )
        ]
    )


@dataclass
class Api:
    """Summary of public api for a class."""

    cls: type[Any]
    description: str
    methods: dict[str, tuple[str, str, set[type[Any]]]]
    attributes: dict[str, tuple[type[Any], str, set[type[Any]]]] = field(
        default_factory=dict
    )
    enum_members: dict[str, tuple[Any, str]] = field(default_factory=dict)
    domain_capabilities: dict[str, type[Any]] = field(default_factory=dict)
    domain_base_types: dict[str, type[Any]] = field(default_factory=dict)
    domain_observation_space_cls: Optional[type[Space]] = None
    domain_action_space_cls: Optional[type[Space]] = None


def format_api_for_prompt(
    api: Api,
    subtitle_level: int = 1,
) -> str:
    """Format api extracted via get_public_apis() for prompt.

    # Parameters
    api
    subtitle_level

    # Returns

    """
    subtitle_prefix = "#" * (subtitle_level - 1)

    # title
    output = [subtitle_prefix + f"# API Reference: `{repr(api.cls)}`\n"]

    # Domain specific: capabilities
    if api.domain_capabilities:
        output.append(
            _format_domain_capabilities_for_prompt(
                api.domain_capabilities, subtitle_level=subtitle_level + 1
            )
        )
    # Domain specific: base types
    if api.domain_base_types:
        output.append(
            _format_domain_base_types_for_prompt(
                api.domain_base_types, subtitle_level=subtitle_level + 1
            )
        )

    # Domain specific: observation and action space classes
    if api.domain_action_space_cls:
        output.append(f"\n{subtitle_prefix}## Action space per agent")
        output.append(f"{repr(api.domain_action_space_cls)}\n")
    if api.domain_observation_space_cls:
        output.append(f"\n{subtitle_prefix}## Observation space per agent")
        output.append(f"{repr(api.domain_observation_space_cls)}\n")

    # Description
    description = api.description
    if description:
        output.append(f"\n{subtitle_prefix}## Description")
        output += _format_docstring_as_paragraph_for_prompt(
            description, subtitle_level=subtitle_level + 2
        )

    # Enum members
    if issubclass(api.cls, Enum):
        enum_members = api.enum_members
        output += [
            f"\n{subtitle_prefix}## Enum members",
        ]
        for name, (value, descr) in enum_members.items():
            line = f"- **{name}** (Value: `{repr(value)}`): "
            if descr:
                line += _format_docstring_as_inline_for_prompt(descr)
            output.append(line)

    # Attributes
    else:
        output.append("\n" + subtitle_prefix + "## Attributes")
        attributes = api.attributes
        if attributes:
            for attr_name, (attr_type, descr, ref_types) in attributes.items():
                line = f"- **{attr_name}** ({attr_type}): "
                if descr:
                    line += _format_docstring_as_inline_for_prompt(descr)
                output.append(line)
        else:
            output.append("None detected")

    # Methods
    output.append("\n" + subtitle_prefix + "## Methods")
    if api.methods:
        for name, (sig, descr, ref_types) in api.methods.items():
            output.append(subtitle_prefix + f"### - `{name}{sig}`")
            if descr:
                output += _format_docstring_as_paragraph_for_prompt(
                    descr, subtitle_level=subtitle_level + 3
                )
            output.append("\n")
    else:
        output.append("None detected")

    prompt_addon = "\n".join(output)
    # Final cleanup of excess whitespace
    prompt_addon = re.sub(r"\n\s*\n", "\n\n", prompt_addon).strip()

    return prompt_addon


def _format_docstring_as_paragraph_for_prompt(
    docstring: str, subtitle_level: int = 1
) -> list[str]:
    subtitle_prefix = "#" * (subtitle_level - 1)
    input_lines = docstring.split("\n")
    output_lines = []
    # handles '#' in the dosctring
    in_code_block = False
    for line in input_lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        if line.startswith("#") and not in_code_block:
            output_lines.append(subtitle_prefix + line)
        else:
            output_lines.append(line)
    return output_lines


def _format_docstring_as_inline_for_prompt(docstring: str) -> str:
    input_lines = docstring.split("\n")
    output_lines = []
    # handles '#' in the dosctring
    in_code_block = False
    for line in input_lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
        if line.startswith("#") and not in_code_block:
            output_lines.append(f"**{line}**:")
        else:
            output_lines.append(line)
    return "\n".join(output_lines)


def get_public_apis(
    classes: list[type[Any]],
    recursive: bool = True,
    domain_cls: Optional[type[Any]] = None,
    domain: Optional[Domain] = None,
    strip_admonitions: bool = False,
    extract_observation_space_cls: bool = True,
    extract_action_space_cls: bool = True,
    simplify_signature: bool = True,
    include_hub_api: bool = True,
    include_domain_cls_module: bool = True,
    user_modules: Optional[list[ModuleType]] = None,
) -> list[Api]:
    """Maps each class to its public api represented as a dict.

    # Parameters
    classes: list of classes to document
    recursive: if True, also document the classes found in method/attribute annotations, that are filtered by `is_class_to_document`
    domain_cls: domain class for which these apis are required
    domain: if provided, this domain instance is used to detect action space and observation space class (in order to expose their api)
    strip_admonitions: whether stripping admonitions like "!!! tip" or "!!! warning" from dosctrings
    extract_observation_space_cls: whether using the domain instance to extract the observation space class
    extract_action_space_cls: whether using the domain instance to extract the action space class
    simplify_signature: whether simplifying method signatures (replacing D.T_agent, T_memory and T_concurrency according to mixins, and flattening Union's)
    include_hub_api: see `is_class_to_document`
    include_domain_cls_module: see `is_class_to_document`
    user_modules: see `is_class_to_document`

    """
    apis: list[Api] = []
    classes_done = set()
    while classes:
        cls = classes.pop(0)
        api, referenced_classes = _get_public_api_and_referenced_types(
            cls,
            domain_cls=domain_cls,
            domain=domain,
            extract_action_space_cls=extract_action_space_cls,
            extract_observation_space_cls=extract_observation_space_cls,
            strip_admonitions=strip_admonitions,
            simplify_signature=simplify_signature,
        )
        apis.append(api)
        classes_done.add(cls)
        if recursive:
            classes += [
                cls
                for cls in referenced_classes
                if (
                    cls not in classes
                    and cls not in classes_done
                    and is_class_to_document(
                        cls,
                        domain_cls=domain_cls,
                        include_hub_api=include_hub_api,
                        include_domain_cls_module=include_domain_cls_module,
                        user_modules=user_modules,
                    )
                )
            ]
    return apis


def _get_public_api_and_referenced_types(
    cls: type[Any],
    domain_cls: Optional[type[Domain]] = None,
    domain: Optional[Domain] = None,
    strip_admonitions: bool = False,
    extract_observation_space_cls: bool = True,
    extract_action_space_cls: bool = True,
    simplify_signature: bool = True,
) -> tuple[Api, list[type[Any]]]:
    ref_types: set[type[Any]] = set()

    description = _sanitize_scikit_docstring(
        cls.__doc__, strip_admonitions=strip_admonitions
    )

    domain_capabilities = {}
    domain_base_types = {}
    observation_space_cls = None
    action_space_cls = None
    if issubclass(cls, Domain):
        domain_capabilities = _extract_domain_characteristics(cls)
        domain_base_types = _extract_domain_base_types(domain_cls)
        ref_types.update(
            {
                simple_type
                for base_type in domain_base_types.values()
                for simple_type in _extract_simple_types_from_nested_type(base_type)
            }
        )
        if domain is not None:
            # Action space and observation space types
            if extract_action_space_cls:
                if isinstance(domain, SingleAgent):
                    action_space_cls = type(domain.get_action_space())
                else:
                    action_space_cls = type(
                        next(iter(domain.get_action_space().values()))
                    )
                ref_types.add(action_space_cls)
            if extract_observation_space_cls:
                if isinstance(domain, SingleAgent):
                    observation_space_cls = type(domain.get_observation_space())
                else:
                    observation_space_cls = type(
                        next(iter(domain.get_observation_space().values()))
                    )
                ref_types.add(observation_space_cls)

    methods = _get_public_methods(
        cls,
        domain_cls=domain_cls,
        simplify_signature=simplify_signature,
        strip_admonitions=strip_admonitions,
    )
    ref_types.update(
        {
            ref_type
            for _, _, method_ref_types in methods.values()
            for ref_type in method_ref_types
        }
    )
    if issubclass(cls, Enum):
        enum_members = _get_enum_members_value_and_doc(
            cls, strip_admonitions=strip_admonitions
        )
        attributes = {}
    else:
        enum_members = {}
        attributes = _get_public_attributes(
            cls, domain_cls=domain_cls, strip_admonitions=strip_admonitions
        )
        ref_types.update(
            {t for _, _, attr_ref_types in attributes.values() for t in attr_ref_types}
        )

    api = Api(
        cls=cls,
        description=description,
        methods=methods,
        attributes=attributes,
        enum_members=enum_members,
        domain_capabilities=domain_capabilities,
        domain_base_types=domain_base_types,
        domain_observation_space_cls=observation_space_cls,
        domain_action_space_cls=action_space_cls,
    )
    # remove string repr and D.T_event and co
    sanitized_ref_types = [t for t in ref_types if isinstance(t, type)]
    return (api, sanitized_ref_types)


def _get_enum_members_value_and_doc(
    cls: type[Enum], strip_admonitions: bool = False
) -> dict[str, tuple[Any, str]]:
    ast_docs = _parse_attributes_doc_with_ast(cls, strip_admonitions=strip_admonitions)
    enm_doc = cls.__doc__
    enum_members_api = {}
    for member in cls:
        name = member.name
        doc = ast_docs.get(name)
        if not doc:
            temp_doc = member.__doc__
            if temp_doc and temp_doc != enm_doc:  # Ensure it's not the class doc
                doc = _sanitize_scikit_docstring(
                    temp_doc, strip_admonitions=strip_admonitions
                )
            else:
                doc = ""
        enum_members_api[name] = (member.value, doc)
    return enum_members_api


def _get_public_methods(
    cls: type[Any],
    domain_cls: Optional[type[Domain]] = None,
    simplify_signature: bool = True,
    strip_admonitions: bool = False,
) -> dict[str, tuple[str, str, set[type[Any]]]]:
    methods = dict()

    def is_method_name_to_doc(name):
        return name == "__init__" or not (name.startswith("_") or name in methods)

    # Dynamic Inspection for methods or class attributes
    for name, method in inspect.getmembers(cls):
        if inspect.isroutine(method):
            if is_method_name_to_doc(name):
                try:
                    # Capture the signature for methods
                    sig = str(inspect.signature(method))
                    if simplify_signature:
                        sig = _simplify_signature_repr(sig, domain_cls=domain_cls)
                except ValueError:
                    sig = "(...)"
                doc = _sanitize_scikit_docstring(
                    method.__doc__, strip_admonitions=strip_admonitions
                )
                try:
                    ref_types = _extract_ref_types_from_method(
                        method=method, domain_cls=domain_cls, drop_D_T_xxx=True
                    )
                except:
                    ref_types = []
                methods[name] = sig, doc, ref_types
    return methods


def _simplify_signature_repr(sig: str, domain_cls: Union[type[Domain], None]) -> str:
    """Simplify method signature.

    Replace D.T_xxx by their counterpart defined in characteristic mixins
    - T_agent -> Union (SingleAgent) or StrDict (Multigent)
    - T_concurrency
    - T_memory

    Remove extra Union having only one argument.

    """
    # Replace D.T_xxx by its value (only the ones predefined by characteristcis mixins)
    if domain_cls:
        predefined_types = _extract_domain_base_types(
            domain_cls, skip_predefined=False, only_predefined=True
        )
        for k, v in predefined_types.items():
            try:
                type_name = v.__name__
            except AttributeError:
                type_name = str(v)
            sig = sig.replace(f"D.{k}", type_name)
    # Remove extra Union[...] if only one arg
    sig = _flatten_unions_in_signature(sig)
    return sig


def _flatten_unions_in_signature(sig: str) -> str:
    def simplify_annotation(annotation: str) -> str:
        # Tokenize: names (including dots), [, ], and commas
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_.]*|[\[\],]", annotation)
        if not tokens:
            return annotation

        def parse(index):
            if index >= len(tokens):
                return "", index
            token = tokens[index]

            # Check for generic type structure: Name[...]
            if index + 1 < len(tokens) and tokens[index + 1] == "[":
                type_name = token
                inner_types, curr = [], index + 2

                while curr < len(tokens) and tokens[curr] != "]":
                    subtree, next_index = parse(curr)
                    inner_types.append(subtree)
                    curr = next_index
                    if curr < len(tokens) and tokens[curr] == ",":
                        curr += 1

                new_index = curr + 1

                if type_name == "Union":
                    # 1. Flatten nested Unions
                    flattened = []
                    for t in inner_types:
                        if t.startswith("Union["):
                            # Extract inner content and split by top-level commas
                            inner_content = t[6:-1]
                            flattened.extend(
                                re.split(r",\s*(?![^\[]*\])", inner_content)
                            )
                        else:
                            flattened.append(t)

                    # 2. Simplify if only one argument remains
                    if len(flattened) == 1:
                        return flattened[0], new_index
                    return f"Union[{', '.join(flattened)}]", new_index

                return f"{type_name}[{', '.join(inner_types)}]", new_index

            return token, index + 1

        try:
            result, _ = parse(0)
            return result
        except:
            return annotation

    def replacement_logic(match):
        prefix = match.group(1)  # e.g., ": " or "-> "
        quote = match.group(2)  # ' or "
        content = match.group(3)  # The actual type string

        simplified = simplify_annotation(content)
        return f"{prefix}{quote}{simplified}{quote}"

    # Specifically matches string literals used as annotations
    # Targets: : 'Type'  OR  -> 'Type'
    pattern = r"(:\s*|->\s*)('|\")([^'\"]+)\2"
    return re.sub(pattern, replacement_logic, sig)


class _fake_D_attr(Generic[T]): ...


def _extract_ref_types_from_method(
    method: typing.Callable[[...], Any],
    domain_cls: Optional[type[Domain]] = None,
    drop_D_T_xxx: bool = True,
) -> set[type[Any]]:
    # special namespace for scikit-decide domains to avoid extracting D.T_...
    localns = _get_local_namespace_for_D_T_xxx(
        domain_cls=domain_cls, drop_D_T_xxx=drop_D_T_xxx
    )

    return {
        simple_type
        for arg_type in get_type_hints(method, localns=localns).values()
        for simple_type in _extract_simple_types_from_nested_type(arg_type)
    }


def _extract_simple_types_from_nested_type(t: type[Any]) -> set[type[Any]]:
    """
    Recursively extracts base types from nested generics.
    Returns a set of type objects.
    Exclude Any, Union, and other special typing types.
    """
    found = set()

    # Handle containers like Union, list, ...
    origin = typing.get_origin(t)
    args = typing.get_args(t)

    if origin is not None:
        # Avoid adding container if Union
        if not isinstance(origin, (typing._SpecialForm)) and origin != _fake_D_attr:
            found.add(origin)

        # Recurse into the arguments (the "inner" types)
        for arg in args:
            found.update(_extract_simple_types_from_nested_type(arg))

    elif t is not None and t is not type(None):
        # "Leaf" type (int, float, MyCustomClass)
        # Filter out Any
        if not isinstance(t, typing._SpecialForm) and t != _fake_D_attr:
            found.add(t)

    return found


def _parse_attributes_doc_with_ast(
    cls: type[Any], strip_admonitions: bool = False
) -> dict[str, str]:
    """Extract class attributes doc with ast
    Parses the class source to map attributes names to the docstrings
    written immediately below them.
    """
    attr_docs = {}
    for supercls in cls.__mro__:
        # iterate through inherited classes for doc (keep first doc found)
        try:
            source = inspect.getsource(supercls)
            tree = ast.parse(textwrap.dedent(source))
            class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef))

            for i, node in enumerate(class_node.body):
                # Handle assignments in class body (Class attributes or Enums)
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    name = ""
                    if isinstance(node, ast.Assign) and isinstance(
                        node.targets[0], ast.Name
                    ):
                        name = node.targets[0].id
                    elif isinstance(node, ast.AnnAssign) and isinstance(
                        node.target, ast.Name
                    ):
                        name = node.target.id

                    if name and name not in attr_docs and i + 1 < len(class_node.body):
                        next_node = class_node.body[i + 1]
                        if isinstance(next_node, ast.Expr) and isinstance(
                            next_node.value, ast.Constant
                        ):
                            attr_docs[name] = _sanitize_scikit_docstring(
                                next_node.value.value.strip(),
                                strip_admonitions=strip_admonitions,
                            )
        except:
            pass
    return attr_docs


def _get_local_namespace_for_D_T_xxx(
    domain_cls: Optional[type[Domain]] = None, drop_D_T_xxx: bool = True
) -> Optional[dict[str, Any]]:
    if domain_cls is not None:
        if drop_D_T_xxx:

            class DFactory:
                def __init__(self, domain_cls: type[Domain]):
                    self.predefined_types = _extract_domain_base_types(
                        domain_cls, skip_predefined=False, only_predefined=True
                    )

                def __getattr__(self, item):
                    return self.predefined_types.get(item, _fake_D_attr)

            localns = {"D": DFactory(domain_cls=domain_cls)}
        else:
            localns = {"D": domain_cls}
    else:
        localns = None
    return localns


def _get_public_attributes(
    cls, domain_cls: Optional[type[Domain]] = None, strip_admonitions: bool = False
) -> dict[str, tuple[type[Any], str, set[type[Any]]]]:
    attributes_unsolved_types: dict[str, Union[type[Any], str]] = dict()
    attributes_solved_types: dict[str, type[Any]] = dict()

    skip_D_T_xxx = issubclass(cls, Domain)
    if domain_cls is None and issubclass(cls, Domain):
        domain_cls = cls

    def is_attribute_name_to_doc(name, attributes):
        return not (
            name.startswith("_")
            or (name in attributes and attributes[name] not in [Any, "Any", "'Any"])
            or (skip_D_T_xxx and name.startswith("T_"))
        )

    # local namespace to translate D.T_event and co
    localns = _get_local_namespace_for_D_T_xxx(domain_cls=domain_cls, drop_D_T_xxx=True)

    # Get resolved types (no strings if possible) to extract next types to document
    try:
        # try to get attribute hints and resolve them into actual types
        attributes_solved_types.update(
            {
                name: solved_type
                for name, solved_type in get_type_hints(cls, localns=localns).items()
                if is_attribute_name_to_doc(name, attributes_solved_types)
            }
        )
    except NameError:
        ...

    # Get unsolved types (potentially string repr) to use for display
    # Handle dataclasses
    if is_dataclass(cls):
        attributes_unsolved_types.update(
            {
                f.name: f.type
                for f in fields(cls)
                if is_attribute_name_to_doc(f.name, attributes_unsolved_types)
            }
        )

    # Get class attributes via annotations
    attributes_unsolved_types.update(
        {
            name: attr_type
            for name, attr_type in get_annotations(cls, eval_str=False).items()
            if is_attribute_name_to_doc(name, attributes_unsolved_types)
        }
    )

    # Dynamic Inspection for class attributes
    for name, value in inspect.getmembers(cls):
        if not inspect.isroutine(value):
            solved_type = type(value)
            if is_attribute_name_to_doc(name, attributes_solved_types):
                attributes_solved_types[name] = solved_type
            if is_attribute_name_to_doc(name, attributes_unsolved_types):
                attributes_unsolved_types[name] = solved_type

    # Static Analysis for Attributes (including those set in methods, in class and superclasses source code)
    for supercls in cls.__mro__:
        try:
            source = inspect.getsource(supercls)
            tree = ast.parse(textwrap.dedent(source))

            for node in ast.walk(tree):
                # We look for function definitions to capture their local signatures
                if isinstance(node, ast.FunctionDef):
                    # Get the live signature of this specific method
                    try:
                        method = getattr(supercls, node.name)
                        try:
                            # resolve string hints into actual types
                            local_vars_solved = get_type_hints(method, localns=localns)
                        except NameError:
                            local_vars_solved = {}
                        # inspect the signature for str representation
                        try:
                            sig = inspect.signature(getattr(supercls, node.name))
                            local_vars_unsolved = {
                                name: p.annotation
                                for name, p in sig.parameters.items()
                                if p.annotation is not inspect.Parameter.empty
                            }
                        except:
                            local_vars_unsolved = {}

                    except:
                        local_vars_solved = {}
                        local_vars_unsolved = {}

                    # Now look for assignments inside this specific function
                    for subnode in ast.walk(node):
                        _extract_ast_node_attribute_name_and_type(
                            subnode,
                            cls=supercls,
                            local_unsolved_hints=local_vars_unsolved,
                            local_solved_hints=local_vars_solved,
                            attributes_unsolved_types=attributes_unsolved_types,
                            attributes_solved_types=attributes_solved_types,
                            is_attribute_name_to_doc=is_attribute_name_to_doc,
                        )

        except (OSError, TypeError, IndentationError):
            pass

    attributes_doc = _parse_attributes_doc_with_ast(
        cls, strip_admonitions=strip_admonitions
    )

    return {
        attr_name: (
            attributes_unsolved_types.get(attr_name, Any),
            attributes_doc.get(attr_name, ""),
            _extract_simple_types_from_nested_type(
                attributes_solved_types.get(attr_name, Any)
            ),
        )
        for attr_name in set(attributes_unsolved_types).union(attributes_solved_types)
    }


def _extract_ast_node_attribute_name_and_type(
    node,
    cls,
    local_unsolved_hints,
    local_solved_hints,
    attributes_unsolved_types,
    attributes_solved_types,
    is_attribute_name_to_doc,
):
    # Handle annotated assignments (e.g., self.x: int = 10)
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
        if isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
            name = node.target.attr
            # Convert the AST node back to a name/type if possible
            unsolved_type = ast.unparse(node.annotation)
            if is_attribute_name_to_doc(name, attributes_unsolved_types):
                attributes_unsolved_types[name] = unsolved_type
            if is_attribute_name_to_doc(name, attributes_solved_types):
                try:
                    solved_type = eval(
                        unsolved_type, vars(sys.modules[cls.__module__]), vars(cls)
                    )
                except:
                    solved_type = Any
                attributes_solved_types[name] = solved_type

    # Handle standard assignments (e.g., self.x = 10)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                name = target.attr
                if target.value.id == "self":
                    if is_attribute_name_to_doc(name, attributes_unsolved_types):
                        attributes_unsolved_types[name] = _extract_node_type(
                            node.value, local_unsolved_hints, attributes_unsolved_types
                        )
                    if is_attribute_name_to_doc(name, attributes_solved_types):
                        attributes_solved_types[name] = _extract_node_type(
                            node.value, local_solved_hints, attributes_solved_types
                        )


def _extract_node_type(
    node,
    local_vars: dict[str, Union[str, type[Any]]],
    attributes: dict[str, Union[str, type[Any]]],
) -> Union[str, type[Any]]:
    default_type = Any
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        # self.y
        return attributes.get(node.attr, default_type)
    elif isinstance(node, ast.Name):
        # x (x being in the above method signature)
        return local_vars.get(node.id, default_type)
    elif isinstance(node, ast.BinOp):
        # a + b => infer from
        left_type = _extract_node_type(node.left, local_vars, attributes)
        if left_type == default_type:
            return _extract_node_type(node.right, local_vars, attributes)
        else:
            return left_type
    # Fallback: Infer type from the value being assigned
    elif isinstance(node, ast.Constant):
        return type(node.value)
    elif isinstance(node, (ast.List, ast.ListComp)):
        return list
    elif isinstance(node, (ast.Dict, ast.DictComp)):
        return dict
    else:
        return default_type


def _sanitize_scikit_docstring(
    doc: str, strip_admonitions: bool = False, reformat_admonitions: bool = True
) -> str:
    if not doc:
        return ""

    endofparagraph_pattern = "\n\n|\n#+\s|\n!!!|$"

    # Remove triple-quotes indentation
    doc = inspect.cleandoc(doc)

    # Format # Parameters and # Attributes as a list
    def listify_params(match):
        header = match.group(1)
        content = match.group(2)

        # Identify 'name: description' pairs.
        # We look for lines starting with a word followed by a colon.
        # Then we capture everything until the next 'name:' or a double newline.
        # This regex handles indented continuation lines.
        param_pattern = r"(^[\w\._]+):"

        lines = content.split("\n")
        formatted_list = []
        current_param = None

        for line in lines:
            if not line.strip():
                continue

            # Check if line starts a new parameter
            param_match = re.match(r"^([\w\._]+):", line.strip())
            if line.startswith(" ") and current_param:
                # This is a continuation line (indented)
                # Append it to the last item in our list
                formatted_list[-1] += " " + line.strip()
            else:
                if ":" not in line:
                    # missing :
                    line = line + ":"
                name, desc = line.split(":", 1)
                name = name.strip()
                desc = desc.strip()
                current_param = f"- **{name}**: {desc}"
                formatted_list.append(current_param)

        return header + "\n".join(formatted_list)

    for section_name in ["Parameters", "Attributes"]:
        param_section_pattern = (
            rf"(# {section_name}\n)(.*?)(?={endofparagraph_pattern})"
        )
        doc = re.sub(param_section_pattern, listify_params, doc, flags=re.DOTALL)

    # Handle Admonition Blocks (!!! warning, !!! tip)
    # This regex captures the type (warning/tip) and the indented content following it
    admonition_pattern = r"!!!\s+(\w+)\s*\n((?:\s.*?(?:\n|$))+)"

    if strip_admonitions:
        doc = re.sub(admonition_pattern, "", doc)
    elif reformat_admonitions:

        def reformat(match):
            label = match.group(1).upper()
            content = match.group(2).replace("    ", "").strip()  # Remove indentation
            return f"**{label}:** {content}\n"

        doc = re.sub(admonition_pattern, reformat, doc)

    # Remove internal paragraphs
    internal_keywords = ["By default,", "It also autocasts", "Internally calls"]
    for kw in internal_keywords:
        # This pattern finds the keyword and eats everything until:
        # - Two newlines (\n\n)
        # - A Markdown header (\n#+\s)
        # - An admonition (\n!!!)
        # - The end of the string ($)
        pattern = rf"{kw}.*?(?={endofparagraph_pattern})"
        doc = re.sub(pattern, "", doc, flags=re.DOTALL | re.IGNORECASE)

    # Strip '#' internal cross-link prefixes (e.g., #Environment.step())
    doc = re.sub(r"#(\w+)", r"\1", doc)

    # Final cleanup of excess whitespace
    doc = re.sub(r"\n\s*\n", "\n\n", doc).strip()

    return doc
