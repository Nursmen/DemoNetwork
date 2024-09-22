import typing as t
from composio.cli.context import Context
from composio.client import Composio, Entity
from composio.client.collections import (
    AppAuthScheme,
    AuthSchemeField,
    AppModel,
    IntegrationModel,
)
from composio.client.exceptions import ComposioClientError
from composio.utils.url import get_web_url
from composio.cli.context import get_context
from composio.cli.connections import _connections

def _load_integration(
    context: Context,
    integration_id: t.Optional[str] = None,
) -> t.Optional[IntegrationModel]:
    
    if integration_id is None:
        return None

    for integration in context.client.integrations.get():
        if integration.id == integration_id:
            return integration

    raise ComposioClientError(f"Integration {integration_id} not found")

def add_integration(
    name: str,
    context: Context = get_context(),
    entity_id: str = 'default',
    integration_id: t.Optional[str] = None,
    no_browser: bool = False,
    auth_mode: t.Optional[str] = None,
    scopes: t.Optional[t.Tuple[str, ...]] = None,
    force: bool = True,
) -> None:
    
    entity = context.client.get_entity(id=entity_id)
    integration = _load_integration(
        context=context,
        integration_id=integration_id,
    )

    try:
        existing_connection = entity.get_connection(app=name)
    except ComposioClientError:
        existing_connection = None

    if existing_connection is not None and not force:
        print(
            f"[yellow]Warning: An existing connection for {name} was found.[/yellow]\n"
        )

        return None

    if existing_connection is not None and force:
        print(
            f"[yellow]Warning: Replacing existing connection for {name}.[/yellow]\n"
        )

    print(
        f"\n[green]> Adding integration: {name.capitalize()}...[/green]\n"
    )

    app = t.cast(AppModel, context.client.apps.get(name=name))

    if app.no_auth:
        raise ComposioClientError(f"{app.name} does not require authentication")

    auth_schemes = app.auth_schemes or []
    if len(auth_schemes) == 0:
        context.console.print(f"{app.name} does not need authentication")
        return None

    auth_modes = {auth_scheme.auth_mode: auth_scheme for auth_scheme in auth_schemes}
    if auth_mode is not None and auth_mode not in auth_modes:
        raise ComposioClientError(
            f"Invalid value for `auth_mode`, select from `{set(auth_modes)}`"
        )

    if auth_mode is not None:
        auth_scheme = auth_modes[auth_mode]
    elif len(auth_modes) == 1:
        ((auth_mode, auth_scheme),) = auth_modes.items()
    else:        
        auth_mode = list(auth_modes)[0]
        auth_scheme = auth_modes[auth_mode]

    if auth_mode.lower() in ("basic", "api_key"):
        return _handle_basic_auth(
            entity=entity,
            client=context.client,
            app_name=name,
            auth_mode=auth_mode,
            auth_scheme=auth_scheme,
        )
    return _handle_oauth(
        entity=entity,
        client=context.client,
        app_name=name,
        no_browser=no_browser,
        integration=integration,
        scopes=scopes,
    )


def _get_auth_config(
    scopes: t.Optional[t.Tuple[str, ...]] = None
) -> t.Optional[t.Dict]:
    
    scopes = scopes or ()
    if len(scopes) == 0:
        return None
    return {"scopes": ",".join(scopes)}


def _handle_oauth(
    entity: Entity,
    client: Composio,
    app_name: str,
    no_browser: bool = False,
    integration: t.Optional[IntegrationModel] = None,
    scopes: t.Optional[t.Tuple[str, ...]] = None,
) -> None:
    
    connection = entity.initiate_connection(
        app_name=app_name.lower(),
        redirect_url=get_web_url(path="redirect"),
        integration=integration,
        auth_mode="OAUTH2",
        auth_config=_get_auth_config(scopes=scopes),
        use_composio_auth=True,
        force_new_integration=len(scopes or []) > 0,
    )
    # if not no_browser:
    #     webbrowser.open(
    #         url=str(connection.redirectUrl),
    #     )
    # click.echo(
    #     f"Please authenticate {app_name} in the browser and come back here. "
    #     f"URL: {connection.redirectUrl}"
    # )
    # click.echo(f"⚠ Waiting for {app_name} authentication...")
    # connection.wait_until_active(client=client)
    # click.echo(f"✔ {app_name} added successfully!")

    return connection.redirectUrl

def _collect_input_fields(
    fields: t.List[AuthSchemeField],
    expected_from_customer: bool = False,
) -> t.Dict:
    
    inputs = {}
    for _field in fields:
        field = _field.model_dump()
        if field.get("expected_from_customer", True) and expected_from_customer:
            if field.get("required", False):
                value = input(
                    f"> Enter {field.get('display_name', field.get('name'))}: "
                )
                if not value:
                    raise ComposioClientError(
                        f"{field.get('display_name', field.get('name'))} is required"
                    )
            else:
                value = input(
                    f"Enter {field.get('display_name', field.get('name'))} (Optional):"
                ) or t.cast(
                    str,
                    field.get("default"),
                )
            inputs[field.get("name")] = value
    return inputs

def _handle_basic_auth(
    entity: Entity,
    client: Composio,
    app_name: str,
    auth_mode: str,
    auth_scheme: AppAuthScheme,
    integration: t.Optional[IntegrationModel] = None,
) -> None:
    
    entity.initiate_connection(
        app_name=app_name.lower(),
        auth_mode=auth_mode,
        auth_config=_collect_input_fields(
            fields=auth_scheme.fields,
            expected_from_customer=True,
        ),
        integration=integration,
        use_composio_auth=False,
        force_new_integration=True,
    ).save_user_access_data(
        client=client,
        field_inputs=_collect_input_fields(
            fields=auth_scheme.fields,
            expected_from_customer=False,
        ),
        entity_id=entity.id,
    )
    
    return "Basic auth added successfully!"

def check_integration(name:str) -> bool:
    context = get_context()

    integrations = context.client.integrations.get()

    return any(integration.appName == name for integration in integrations)

if __name__ == "__main__":
    print(check_integration('weathermap'))

    print(add_integration('weathermap'))